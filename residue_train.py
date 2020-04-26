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

from helper import psnr, load_imgs
from coarse_test import coarse16_test
from prediction_inference import pred_inference

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

def regroup(N_frames, images_shape, bm, predicted_frames):
    
    width, height = images_shape[1], images_shape[2]
    
    final_prediction=[]
    
    i = 0
    for n in range(N_frames - 2):
        result = np.zeros((width, height, 3))
        
        for y in range(0, width, bm):
           for x in range(0, height, bm):
               res = x
               result[y:y + bm, x:x + bm,:] = predicted_frames[i].reshape(bm,bm,3)
               i = i + 1
              
        final_prediction.append(result)
    
    final_prediction = np.array(final_prediction) # re-group the decoded frames
        
    return final_prediction
      
def residue_train(folder, start, end, bm, b, pred):
    N_frames = end-start # including the ref. frames

    images = load_imgs(folder, start+1, end-1)
    
    print(images.shape)

    width, height = images.shape[1], images.shape[2]
    
   
    residue = images - pred
    print(residue.shape)
    
    C = []
    
    for i in range(0, N_frames-2): 
        current = residue[i]
        for y in range(0, width, b):
            for x in range(0, height, b):
                block = current[y:y + b, x:x + b]
                block = block.reshape(b*b, 3)
                C.append(block)
    
    C = np.array(C)
    C = C.reshape(C.shape[0], b, b, 3)
    print(C.shape)
    
    input1 = Input(shape = (b, b, 3))
    
    y = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = (4, 4), activation='relu')(input1)
    y = Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    y = Conv2D(3, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)  
  
    y = Conv2DTranspose(64, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    y = Conv2DTranspose(128, kernel_size=(5, 5), padding = "SAME", strides = 1, activation='relu')(y)
    y = Conv2DTranspose(3, kernel_size=(5, 5), padding = "SAME", strides = (4, 4), activation='relu')(y)
    
    residue_model = Model(inputs = input1, outputs = y)
    
    residue_model.summary()
    residue_model.compile(optimizer='adam', loss='mse')
  
 # ============== YL ===============================
    # save model
   
    model_json = residue_model.to_json()
    with open("./models/BlowingBubbles_416x240_50_residue16.json", "w") as json_file:
        json_file.write(model_json)

    
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=2, patience=10, \
                              verbose=2, mode='auto', \
                              baseline=None, restore_best_weights=True)                    
    # define modelcheckpoint callback
    checkpointer = ModelCheckpoint(filepath='./models/BlowingBubbles_416x240_50_residue16.hdf5',\
                                   monitor='val_loss',save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    residue_model.fit(C, C, batch_size=100, epochs=100, verbose=2, validation_split=0.2, callbacks=callbacks_list)
    
if __name__ == "__main__":   
    folder = './dataset/BlowingBubbles_416x240_50/'
    
    b = 16 # blk_size & ref. blk size
    test_start, test_end = 100, 200
    train_start, train_end = 0, 20

    images =  load_imgs(folder, train_start, train_end)
    coarse_frames = coarse16_test(images, b)
    bm = 8 # target block size to predict

    N_frames = train_end - train_start
    predicted_frames = pred_inference(N_frames, b, bm, images, coarse_frames)
    final_prediction = regroup(N_frames, images.shape, bm, predicted_frames)
 
    residue_train(folder, train_start, train_end, bm, b, final_prediction)
    