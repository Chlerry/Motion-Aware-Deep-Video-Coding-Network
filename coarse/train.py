# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utility.helper import psnr, load_imgs, image_to_block, add_noise
from utility.parameter import *

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

def model(images, b, ratio, mode = 'default'):

    coarse_train_set = image_to_block(images, b)
    
    # DL: Load stride and channel from utility.parameter.get_strides_channel
    channel, strides0 = get_channel_strides(ratio)
    
    input_coarse = Input(shape = (b, b, 3))
    
    e = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = strides0, activation='relu', input_shape=(b, b, 3))(input_coarse)
    e = Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    e = Conv2D(channel, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    
    if(mode == 'noise'):
        e = Lambda(add_noise)(e)

    d = Conv2DTranspose(64, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    d = Conv2DTranspose(128, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(d)
    d = Conv2DTranspose(3, kernel_size=(5, 5), padding = "SAME", strides = strides0, activation='relu')(d)
    
    coarse_model = Model(inputs = input_coarse, outputs = d)
    coarse_model.summary()
    
    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    coarse_model.compile(optimizer=opt, loss=keras.losses.MeanAbsoluteError()) # RK

    # ============== DL ===============================
    json_path, hdf5_path = get_model_path("coarse", ratio)
    delta, n_patience, batch_size, epoch_size = get_training_parameter("coarse")
    
    # ============== YL ===============================
    # save model and load model: serialize model to JSON
    model_json = coarse_model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=delta, patience=n_patience, \
                              verbose=2, mode='min', \
                              baseline=None, restore_best_weights=True)                    
    # define modelcheckpoint callback
    checkpointer = ModelCheckpoint(filepath = hdf5_path,\
                                   monitor = 'val_loss', save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    coarse_model.fit(coarse_train_set, coarse_train_set, batch_size=batch_size, epochs=epoch_size, verbose=2, validation_split=0.2, callbacks=callbacks_list)
    # ===================================================

def main(args = 1): 
    b = 16 

    train_images = load_imgs(data_dir, train_start, train_end) 
    
    model(train_images, b, training_ratio, 'noise')

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])