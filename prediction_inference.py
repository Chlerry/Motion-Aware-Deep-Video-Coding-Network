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

def pred_inference(N_frames, b, bm, images, coarse_frames):
        
    width, height = images.shape[1], images.shape[2]

    N_blocks = int((width*height)/(b*b))
    
    decoded = []
    f = coarse_frames.reshape(N_frames, N_blocks, b*b, 3)
    for n in f:
        result = np.zeros((width, height, 3))
        i = 0
        for y in range(0, result.shape[0], b):
           for x in range(0, result.shape[1], b):
               res = n[i].reshape(b,b, 3)
               result[y:y + b, x:x + b] = res
               i = i + 1
              
        decoded.append(result)
    
    decoded = np.array(decoded) # re-group the decoded frames
    
    prev = []
    
    for i in range(0, N_frames-2): # take the reference frames
        img = decoded[i] # frame the decoded frames
        
        for y in range(0, img.shape[0], bm):
            for x in range(0, img.shape[1], bm):
                block = np.zeros((b, b, 3))
                if (y + b) >= img.shape[0] and (x + b) < img.shape[1] :
                    block = img[y-bm:y-bm+b, x:x+b]
                    block = block.reshape(b*b,3)
                    prev.append(block)
                elif (x + b) >= img.shape[1] and (y + b) < img.shape[0]:
                    block = img[y:y+b, x-bm:x-bm+b]
                    block = block.reshape(b*b, 3)
                    prev.append(block)
                elif (y + b) >= img.shape[0] and (x + b) >= img.shape[1] :
                    block = img[y-bm:y-bm+b, x-bm:x-bm+b]
                    block = block.reshape(b*b, 3)
                    prev.append(block)
                else:
                    block = img[y:y + b, x:x + b]
                    block = block.reshape(b*b, 3)
                    prev.append(block)
    
                
    prev = np.array(prev)
    prev = prev.reshape(prev.shape[0], b, b, 3)
    print(prev.shape)
    
    B = []
    
    for i in range(0, N_frames-2): 
        next_f = decoded[i+2]
        
        for y in range(0, img.shape[0], bm):
            for x in range(0, img.shape[1], bm):
                if (y + b) >= img.shape[0] and (x + b) < img.shape[1] :
                    block = next_f[y-bm:y-bm+b, x:x+b]
                    block = block.reshape(b*b,3)
                    B.append(block)
                elif (x + b) >= img.shape[1] and (y + b) < img.shape[0]:
                    block = next_f[y:y+b, x-bm:x-bm+b]
                    block = block.reshape(b*b, 3)
                    B.append(block)
                elif (y + b) >= img.shape[0] and (x + b) >= img.shape[1] :
                    block = next_f[y-bm:y-bm+b, x-bm:x-bm+b]
                    block = block.reshape(b*b, 3)
                    B.append(block)
                else:
                    block = next_f[y:y + b, x:x + b]
                    block = block.reshape(b*b, 3)
                    B.append(block)
    

    B = np.array(B)
    B = B.reshape(B.shape[0], b, b, 3)
    print(B.shape)
    
    C = []
    
    for i in range(0, N_frames-2): 
        current = images[i+1]
        for y in range(0, img.shape[0], bm):
            for x in range(0, img.shape[1], bm):
                block = current[y:y + bm, x:x + bm]
                block = block.reshape(bm*bm, 3)
                C.append(block)
    
    C = np.array(C)
    C = C.reshape(C.shape[0], bm, bm, 3)
    print(C.shape)
    
    # ============== YL: load model ===============================
    
    from keras.models import model_from_json
    json_file = open('./models/BlowingBubbles_416x240_50_pred16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pred_model = model_from_json(loaded_model_json)
    # load weights into new model
    pred_model.load_weights("./models/BlowingBubbles_416x240_50_pred16.hdf5")
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    pred_model.compile(optimizer='adam', loss='mse', metrics=['acc'])

    predicted_frames = pred_model.predict([prev, B])
    return predicted_frames
    # ===================================================
    
if __name__ == "__main__":   
    
    folder = './dataset/BlowingBubbles_416x240_50/'
    b = 16 # blk_size & ref. blk size
    test_start, test_end = 100, 120
    
    images = load_imgs(folder, test_start, test_end)
    coarse_frames = coarse16_test(images, b)
    bm = 8 # target block size to predict

    N_frames = test_end - test_start
    predicted_frames = pred_inference(N_frames, b, bm, images, coarse_frames)
