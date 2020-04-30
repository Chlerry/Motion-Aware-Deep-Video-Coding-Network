#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:31:07 2020

@author: yingliu
"""

import cv2
import math
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim as ssim
from PIL import Image
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


from helper import psnr, load_imgs, get_block_set, regroup
from coarse_test import coarse16_test
from prediction_inference_b1 import pred_inference_b1

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
# =================================================

def pred_train_b23(N_frames, b, bm, coarse_frames, final_prediction, images):

    width, height = images.shape[1], images.shape[2]

    N_blocks = int((width*height)/(b*b))
    
    prev_decoded = []
    f = coarse_frames.reshape(N_frames, N_blocks, b*b, 3)
    for n in f:
        result = np.zeros((width, height, 3))
        i = 0
        for y in range(0, result.shape[0], b):
           for x in range(0, result.shape[1], b):
               res = n[i].reshape(b,b, 3)
               result[y:y + b, x:x + b] = res
               i = i + 1
              
        prev_decoded.append(result)
    
    prev_decoded = np.array(prev_decoded[:N_frames - 4]) # re-group the prev_decoded frames
    
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    # ============== DL ===============================
    prev = get_block_set(N_frames-4, prev_decoded, b, bm, 0)
    print(prev.shape)
    
    B = get_block_set(N_frames-4, final_prediction, b, bm, 0)
    print(B.shape)
    # =================================================

    C = []
    
    for i in range(1, N_frames - 3): 
        current = images[i+1]
        for y in range(0, images[0].shape[0], bm):
            for x in range(0, images[0].shape[1], bm):
                block = current[y:y + bm, x:x + bm]
                block = block.reshape(bm*bm, 3)
                C.append(block)
    
    C = np.array(C)
    C = C.reshape(C.shape[0], bm, bm, 3)
    print(C.shape)
    

    input1 = Input(shape = (b, b, 3))
    
    y = Conv2D(8, kernel_size=(5, 5), padding = "SAME", strides = 2, activation='relu')(input1)
    y = Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    y = Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    y = Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    y = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    y = Model(inputs = input1, outputs = y)
    
    input2 = Input(shape = (b, b, 3))
    
    x = Conv2D(8, kernel_size=(5, 5), padding = "SAME", strides = 2, activation='relu')(input2)
    x = Conv2D(16, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x)
    x = Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x)
    x = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(x)
    x = Model(inputs = input2, outputs = x)
    
    c = keras.layers.concatenate([y.output, x.output])
    
    z = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(c)
    z = Conv2D(256, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(z)
    z = Conv2D(3, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(z)
    
    pred_model = Model(inputs = [y.input, x.input], outputs = z)
    pred_model.summary()
    pred_model.compile(optimizer='adam', loss='mse')
    
    #pred_model.fit([prev, B], C, epochs=5, batch_size=1)
    
    #pred_model.save_weights('pred_NM_w5.h5')
    # ============== YL ===============================
    # save model and load model
    # from keras.models import model_from_json
    # serialize model to JSON
    model_json = pred_model.to_json()
    with open("./models/BlowingBubbles_416x240_50_pred12_b23.json", "w") as json_file:
        json_file.write(model_json)

    
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=2, patience=10, \
                              verbose=2, mode='auto', \
                              baseline=None, restore_best_weights=True)                    
    # define modelcheckpoint callback
    checkpointer = ModelCheckpoint(filepath='./models/BlowingBubbles_416x240_50_pred12_b23.hdf5',\
                                   monitor='val_loss',save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    pred_model.fit([prev, B], C, batch_size=1000, epochs=200, verbose=2, validation_split=0.2, callbacks=callbacks_list)
    # ===================================================

if __name__ == "__main__":   
    
    folder = './dataset/BlowingBubbles_416x240_50/'
    b = 16 # blk_size & ref. blk size
    train_start, train_end = 0, 100
    
    images = load_imgs(folder, train_start, train_end)
    coarse_frames = coarse16_test(images, b)
    bm = 8 # target block size to predict

    N_frames = train_end - train_start
    predicted_frames = pred_inference_b1(N_frames, b, bm, images.shape, coarse_frames)
    final_prediction = regroup(N_frames - 4, images.shape, bm, predicted_frames)
    print(predicted_frames.shape)
    print(final_prediction.shape)

    pred_train_b23(N_frames, b, bm, coarse_frames, final_prediction, images)
