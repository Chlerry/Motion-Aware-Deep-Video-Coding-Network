import numpy as np
from keras.layers import * # YL
from keras.models import Model


from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from utility.helper import psnr, load_imgs, get_coarse_set
from utility.parameter import *

# ============== DL ===============================
# Limit GPU memory(VRAM) usage in TensorFlow 2.0
# https://github.com/tensorflow/tensorflow/issues/34355
# https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528

import tensorflow as tf

# ---- Method 1 ----
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ---- Method 2 ----
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     except RuntimeError as e:
#         print(e)
# =================================================

def model8(images, b):

    coarse_train_set = get_coarse_set(images, b)
    
    input_coarse = Input(shape = (b, b, 3))
    
    e = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = (2,4), activation='relu', input_shape=(b, b, 3))(input_coarse)
    print(e.shape)
    #e = MaxPool2D(strides=(2,2))(e)
    # stride (1,1)
    e = Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    print(e.shape)
    #e = MaxPool2D(strides=(2,2))(e)
    e = Conv2D(3, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    
    d = Conv2DTranspose(64, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    d = Conv2DTranspose(128, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(d)
    d = Conv2DTranspose(3, kernel_size=(5, 5), padding = "SAME", strides = (2, 4), activation='relu')(d)
    
    coarse_model = Model(inputs = input_coarse, outputs = d)
    coarse_model.summary()
    coarse_model.compile(optimizer='adam', loss='mse') # RK
    
    # ============== YL ===============================
    # save model and load model
    # from keras.models import model_from_json
    # serialize model to JSON
    model_json = coarse_model.to_json()
    with open(corse_train8_json, "w") as json_file:
        json_file.write(model_json)

    
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=2, patience=10, \
                              verbose=2, mode='auto', \
                              baseline=None, restore_best_weights=True)                    
    # define modelcheckpoint callback
    checkpointer = ModelCheckpoint(filepath = corse_train8_hdf5,\
                                   monitor = 'val_loss', save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    coarse_model.fit(coarse_train_set, coarse_train_set, batch_size=coarse_batch_size, epochs=coarse_epoch, verbose=2, validation_split=0.2, callbacks=callbacks_list)
    # ===================================================

if __name__ == "__main__":   
    data_dir = './dataset/BlowingBubbles_416x240_50/'
    b = 16 
    train_start, train_end = 0, 100

    train_images = load_imgs(data_dir, train_start, train_end) 
    model8(train_images, b)