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


from helper import psnr, load_imgs, get_block_set
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

def pred_inference_b1(N_frames, b, bm, images_shape, decoded):
    
    # ============== DL ===============================
    prev = get_block_set(N_frames - 4, decoded, b, bm, 0)
    # print(prev.shape)
    
    B = get_block_set(N_frames - 4, decoded, b, bm, 4)
    # print(B.shape)
    
    # ============== YL: load model ===============================
    
    from keras.models import model_from_json
    json_file = open('./models/BlowingBubbles_416x240_50_pred16_b1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pred_model = model_from_json(loaded_model_json)
    # load weights into new model
    pred_model.load_weights("./models/BlowingBubbles_416x240_50_pred16_b1.hdf5")
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    pred_model.compile(optimizer='adam', loss='mse', metrics=['acc'])

    predicted_frames = pred_model.predict([prev, B])
    return predicted_frames
    # ===================================================
    
if __name__ == "__main__":   
    
    folder = './dataset/BlowingBubbles_416x240_50/'
    b = 16 # blk_size & ref. blk size
    test_start, test_end = 100, 200
    
    images = load_imgs(folder, test_start, test_end)
    coarse_frames = coarse16_test(images, b)
    bm = 8 # target block size to predict

    N_frames = test_end - test_start
    predicted_frames = pred_inference_b1(N_frames, b, bm, images.shape, coarse_frames)
    # print(predicted_frames.shape)