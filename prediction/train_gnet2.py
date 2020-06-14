"""
Created on Wed Feb 12 23:31:07 2020

@author: yingliu
"""
# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from utility.parameter import *
from utility.helper import psnr, load_imgs, get_block_set, regroup, image_to_block

# ============== DL ===============================
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
# ============== DL ===============================
import keras.backend as K
if rtx_optimizer == True:
    K.set_epsilon(1e-4) 
# =================================================

def model(images, decoded, b, bm, ratio):
    
    N_frames = images.shape[0]
    # ============== DL ===============================
    prev = image_to_block(images[:-2], b, True)

    B = image_to_block(images[2:], b, True)
    
    C = image_to_block(decoded[1:-1], bm)
    # ==================================================================================================

    input1 = Input(shape = (b, b, 3))

    y = Conv2D(8, kernel_size=(5, 5), padding = "SAME", strides = 2, activation='relu')(input1)
    y = Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    y = Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    # =================================================
    y0a = Conv2D(24, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(y)
    y0a = Model(inputs = input1, outputs = y0a)

    y0b = Conv2D(8, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(y)
    y0b = Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y0b)
    y0b = Model(inputs = input1, outputs = y0b)

    y0c = Conv2D(8, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(y)
    y0c = Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y0c)
    y0c = Model(inputs = input1, outputs = y0c)

    y0d = MaxPool2D(pool_size=(3, 3), padding = "SAME", strides = 1)(y)
    y0d = Conv2D(8, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(y0d)
    y0d = Model(inputs = input1, outputs = y0d)

    yc0 = keras.layers.concatenate([y0a.output, y0b.output, y0c.output, y0d.output])
    # =================================================

    y1a = Conv2D(48, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(yc0)
    y1a = Model(inputs = input1, outputs = y1a)

    y1b = Conv2D(16, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(yc0)
    y1b = Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y1b)
    y1b = Model(inputs = input1, outputs = y1b)

    y1c = Conv2D(16, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(yc0)
    y1c = Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y1c)
    y1c = Model(inputs = input1, outputs = y1c)

    y1d = MaxPool2D(pool_size=(3, 3), padding = "SAME", strides = 1)(yc0)
    y1d = Conv2D(16, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(y1d)
    y1d = Model(inputs = input1, outputs = y1d)

    yc1 = keras.layers.concatenate([y1a.output, y1b.output, y1c.output, y1d.output])
    yc1 = Model(inputs = input1, outputs = yc1)
    
    # ==================================================================================================
    input2 = Input(shape = (b, b, 3))

    x = Conv2D(8, kernel_size=(5, 5), padding = "SAME", strides = 2, activation='relu')(input2)
    x = Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x)
    x = Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x)
    # =================================================
    x0a = Conv2D(24, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(x)
    x0a = Model(inputs = input2, outputs = x0a)

    x0b = Conv2D(8, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(x)
    x0b = Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x0b)
    x0b = Model(inputs = input2, outputs = x0b)

    x0c = Conv2D(8, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(x)
    x0c = Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x0c)
    x0c = Model(inputs = input2, outputs = x0c)

    x0d = MaxPool2D(pool_size=(3, 3), padding = "SAME", strides = 1)(x)
    x0d = Conv2D(8, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(x0d)
    x0d = Model(inputs = input2, outputs = x0d)

    xc0 = keras.layers.concatenate([x0a.output, x0b.output, x0c.output, x0d.output])
    # =================================================

    x1a = Conv2D(48, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(xc0)
    x1a = Model(inputs = input2, outputs = x1a)

    x1b = Conv2D(16, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(xc0)
    x1b = Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x1b)
    x1b = Model(inputs = input2, outputs = x1b)

    x1c = Conv2D(16, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(xc0)
    x1c = Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x1c)
    x1c = Model(inputs = input2, outputs = x1c)

    x1d = MaxPool2D(pool_size=(3, 3), padding = "SAME", strides = 1)(xc0)
    x1d = Conv2D(16, kernel_size=(1, 1), padding = "SAME", strides = 1, activation='relu')(x1d)
    x1d = Model(inputs = input2, outputs = x1d)

    xc1 = keras.layers.concatenate([x1a.output, x1b.output, x1c.output, x1d.output])
    xc1 = Model(inputs = input2, outputs = xc1)
    
    # ==================================================================================================
    c = keras.layers.concatenate([yc1.output, xc1.output])
    
    z = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(c)
    z = Conv2D(256, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(z)
    z = Conv2D(3, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(z)
    
    pred_model = Model(inputs = [yc1.input, xc1.input], outputs = z)
    
    pred_model.summary()
    # ============== DL ===============================
    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    pred_model.compile(optimizer=opt, loss='mse')
    
    # ============== DL ===============================
    json_path, hdf5_path = get_model_path("prediction_gnet2", ratio)

    delta, n_patience, batch_size, epoch_size = get_training_parameter("prediction_gnet2")
    # ============== YL ===============================
    # save model and load model:serialize model to JSON
    model_json = pred_model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=delta, \
                              patience=n_patience, \
                              verbose=2, mode='min', \
                              baseline=None, restore_best_weights=True)                    
    # define modelcheckpoint callback
    checkpointer = ModelCheckpoint(filepath=hdf5_path, \
                                   monitor='val_loss',save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    pred_model.fit([prev, B], C, batch_size=batch_size, epochs=epoch_size, verbose=2, validation_split=0.2, callbacks=callbacks_list)
    # ===================================================

def main(args = 1):  
    
    b = 16 # blk_size
    bm = 8 # target block size to predict

    train_images = load_imgs(data_dir, train_start, train_end) 

    model(train_images, train_images, b, bm, training_ratio)

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])