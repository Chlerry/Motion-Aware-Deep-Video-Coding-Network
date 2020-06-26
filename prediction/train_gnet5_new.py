"""
Created by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 
"""
# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from utility.parameter import *
from utility.helper import psnr, load_imgs, regroup, image_to_block

# =================================================
# Limit GPU memory(VRAM) usage in TensorFlow 2.0
# https://github.com/tensorflow/tensorflow/issues/34355
# https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# =================================================
import keras.backend as K
if rtx_optimizer == True:
    K.set_epsilon(1e-4) 
# =================================================

def model(images, decoded, b, bm, ratio):
    
    # =================================================
    prev = image_to_block(decoded[:-2], b, True)

    B = image_to_block(decoded[2:], b, True)
    
    C = image_to_block(images[1:-1], bm)
    # ==================================================================================================

    input1 = Input(shape = (b, b, 3))

    y = BatchNormalization()(Conv2D(8, kernel_size=(5, 5), padding = "SAME", strides = 2, activation='relu')(input1))
    # =================================================
    y0b = BatchNormalization()(Conv2D(6, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y))
    y0b = Model(inputs = input1, outputs = y0b)

    y0c = BatchNormalization()(Conv2D(10, kernel_size=(3, 3), padding = "SAME", strides = 1, activation='relu')(y))
    y0c = Model(inputs = input1, outputs = y0c)

    yc0 = keras.layers.concatenate([y0b.output, y0c.output])
    # =================================================
    y1b = BatchNormalization()(Conv2D(12, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(yc0))
    y1b = Model(inputs = input1, outputs = y1b)

    y1c = BatchNormalization()(Conv2D(20, kernel_size=(3, 3), padding = "SAME", strides = 1, activation='relu')(yc0))
    y1c = Model(inputs = input1, outputs = y1c)

    yc1 = keras.layers.concatenate([y1b.output, y1c.output])
    # =================================================
    yc1 = BatchNormalization()(Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(yc1))
    yc1 = BatchNormalization()(Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(yc1))
    yc1 = Model(inputs = input1, outputs = yc1)
    
    # ==================================================================================================
    input2 = Input(shape = (b, b, 3))

    x = BatchNormalization()(Conv2D(8, kernel_size=(5, 5), padding = "SAME", strides = 2, activation='relu')(input2))
    # =================================================
    x0b = BatchNormalization()(Conv2D(6, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x))
    x0b = Model(inputs = input2, outputs = x0b)

    x0c = BatchNormalization()(Conv2D(10, kernel_size=(3, 3), padding = "SAME", strides = 1, activation='relu')(x))
    x0c = Model(inputs = input2, outputs = x0c)

    xc0 = keras.layers.concatenate([x0b.output, x0c.output])
    # =================================================
    x1b = BatchNormalization()(Conv2D(12, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(xc0))
    x1b = Model(inputs = input2, outputs = x1b)

    x1c = BatchNormalization()(Conv2D(20, kernel_size=(3, 3), padding = "SAME", strides = 1, activation='relu')(xc0))
    x1c = Model(inputs = input2, outputs = x1c)

    xc1 = keras.layers.concatenate([x1b.output, x1c.output])
    # =================================================
    xc1 = BatchNormalization()(Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(xc1))
    xc1 = BatchNormalization()(Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(xc1))
    xc1 = Model(inputs = input2, outputs = xc1)

    # ==================================================================================================
    c = keras.layers.concatenate([yc1.output, xc1.output])
    
    z = BatchNormalization()(Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(c))
    z = BatchNormalization()(Conv2D(256, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(z))
    z = BatchNormalization()(Conv2D(3, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(z))
    
    pred_model = Model(inputs = [yc1.input, xc1.input], outputs = z)
    
    pred_model.summary()
    # =================================================
    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    pred_model.compile(optimizer=opt, loss=keras.losses.MeanAbsoluteError())
    
    json_path, hdf5_path = get_model_path("prediction_gnet5", ratio)

    delta, n_patience, batch_size, epoch_size = get_training_parameter("prediction_gnet5")

    # Save model and load model:serialize model to JSON
    model_json = pred_model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    
    # Define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=delta, \
                              patience=n_patience, \
                              verbose=2, mode='min', \
                              baseline=None, restore_best_weights=True)                    
    # Define modelcheckpoint callback
    checkpointer = ModelCheckpoint(filepath=hdf5_path, \
                                   monitor='val_loss',save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    pred_model.fit([prev, B], C, batch_size=batch_size, epochs=epoch_size, verbose=2, validation_split=0.2, callbacks=callbacks_list)
    # ===================================================

def main(args = 1):  
    
    b = 16 
    bm = 8 

    train_images = load_imgs(data_dir, train_start, train_end) 

    model(train_images, train_images, b, bm, training_ratio)

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])