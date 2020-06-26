"""
Refactored and updated by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 

Initially created by Ying Liu on Wed Feb 12 23:31:07 2020
"""
# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Lambda
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from utility.parameter import *
from utility.helper import psnr, load_imgs, regroup, image_to_block, add_noise
import coarse.test, prediction.inference

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
      
def model(residue, b, ratio, mode = 'noise'):
    
    C = image_to_block(residue, b)

    # Load stride and channel from utility.parameter.get_strides_channel
    channel, strides0 = get_channel_strides(ratio)

    input1 = Input(shape = (b, b, 3))
    
    e = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = strides0, activation='relu')(input1)
    e = Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(e)
    e = Conv2D(channel, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(e)  

    if(mode == 'noise'):
        e = Lambda(add_noise)(e)
  
    d = Conv2DTranspose(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(e)
    d = Conv2DTranspose(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(d)
    d = Conv2DTranspose(3, kernel_size=(5, 5), padding = "SAME", strides = strides0, activation='relu')(d)
    
    residue_model = Model(inputs = input1, outputs = d)
    
    residue_model.summary()

    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    residue_model.compile(optimizer=opt, loss='mse')
    
    # =================================================
    json_path, hdf5_path = get_model_path("residue", ratio)
    delta, n_patience, batch_size, epoch_size = get_training_parameter("residue")

    # Save model
    model_json = residue_model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    
    # Define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=delta, patience=n_patience, \
                              verbose=2, mode='min', \
                              baseline=None, restore_best_weights=True)         
    # Define modelcheckpoint callback
    checkpointer = ModelCheckpoint(filepath=hdf5_path, \
                                   monitor='val_loss',save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    residue_model.fit(C, C, batch_size=batch_size, epochs=epoch_size, \
        verbose=2, validation_split=0.2, callbacks=callbacks_list)
    
def main(args = 1): 
    b = 16 
    bm = 8 

    train_images = load_imgs(data_dir, train_start, train_end)
    decoded = coarse.test.predict(train_images, b, training_ratio)

    regrouped_prediction = prediction.inference.predict(decoded, b, bm, training_ratio)
    
    residue_train_images = train_images[1:n_train_frames-1]
    residue = residue_train_images - regrouped_prediction
    model(residue, b, training_ratio)
    
if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])