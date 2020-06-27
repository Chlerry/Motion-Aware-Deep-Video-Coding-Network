"""
Refactored and updated by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 

Initially created by Ying Liu on Wed Feb 12 23:31:07 2020
"""
# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization
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

def model(images, decoded, b, bm, ratio, model = "prediction"):
    
    # =================================================
    prev = image_to_block(decoded[:-2], b, True)

    B = image_to_block(decoded[2:], b, True)
    
    C = image_to_block(images[1:-1], bm)
    # =================================================

    input1 = Input(shape = (b, b, 3))
    
    y = BatchNormalization()(Conv2D(8, kernel_size=(5, 5), padding = "SAME", strides = 2, activation='relu')(input1))
    y = BatchNormalization()(Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y))
    y = BatchNormalization()(Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y))
    y = BatchNormalization()(Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y))
    y = BatchNormalization()(Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y))
    y = Model(inputs = input1, outputs = y)
    
    input2 = Input(shape = (b, b, 3))
    
    x = BatchNormalization()(Conv2D(8, kernel_size=(5, 5), padding = "SAME", strides = 2, activation='relu')(input2))
    x = BatchNormalization()(Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x))
    x = BatchNormalization()(Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x))
    x = BatchNormalization()(Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x))
    x = BatchNormalization()(Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x))
    x = Model(inputs = input2, outputs = x)
    
    c = keras.layers.concatenate([y.output, x.output])
    
    z = BatchNormalization()(Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(c))
    z = BatchNormalization()(Conv2D(256, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(z))
    z = BatchNormalization()(Conv2D(3, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(z))
    
    pred_model = Model(inputs = [y.input, x.input], outputs = z)
    pred_model.summary()
    # =================================================
    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    pred_model.compile(optimizer=opt, loss=keras.losses.MeanAbsoluteError())
    
    # =================================================
    json_path, hdf5_path = get_model_path(model, ratio)

    delta, n_patience, batch_size, epoch_size = get_training_parameter(model)
    
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
    
    b = 16 # blk_size
    bm = 8 # target block size to predict

    train_images = load_imgs(data_dir, train_start, train_end) 
    import coarse.test
    decoded = coarse.test.predict(train_images, b, training_ratio)

    model(train_images, decoded, b, bm, training_ratio)

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])