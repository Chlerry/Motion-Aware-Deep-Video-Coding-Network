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
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from helper import psnr, load_imgs, regroup
from coarse_test import coarse16_test
from prediction_inference import pred_inference
from prediction_inference_b1 import pred_inference_b1
from prediction_inference_b23 import pred_inference_b23

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

def performance_evaluation(N_frames, images,finalpred):
    width, height = images.shape[1], images.shape[2]
    pixel_max = 255.0
    mse = []
    psnr = []
    ssims = []
   
    for i in range(0,N_frames-2):
        img = np.array(images[i].reshape(width*height, 3), dtype = float)
        res = finalpred[i].reshape(width*height, 3)
        m = mean_squared_error(img, res)
        s = ssim(img, res, multichannel=True)
        p = 20 * math.log10( pixel_max / math.sqrt( m ))
        psnr.append(p)
        mse.append(m)
        ssims.append(s)
 
    
    amse = np.mean(mse)
    apsnr = np.mean(psnr)
    assim = np.mean(ssims)
    return amse, apsnr, assim

def residue_inference(N_frames, images, pred, folder_save, json_dir, hdf5_dir): # start

    width, height = images.shape[1], images.shape[2]
    
    residue = images - pred
    
    C = []
    
    for i in range(0, N_frames): 
        current = residue[i]
        for y in range(0, width, b):
            for x in range(0, height, b):
                block = current[y:y + b, x:x + b]
                block = block.reshape(b*b, 3)
                C.append(block)
    
    C = np.array(C)
    C = C.reshape(C.shape[0], b, b, 3)
    # print(C.shape)
    
# load residue16 model
    from keras.models import model_from_json

    json_file = open(json_dir, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    residue_model = model_from_json(loaded_model_json)
    # load weights into new model

    residue_model.load_weights(hdf5_dir)
    print("Loaded model from disk")
    
    residue_decoded = residue_model.predict([C])
    
    final2 = regroup(N_frames, images.shape, b, residue_decoded)
    
    finalpred = np.add(pred, final2)
  
    
    # j = start+1
    # for result in finalpred:
    #     filename = folder_save+str(j)+'.png'
    #     im_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
    #     im = Image.fromarray(im_rgb)
    #     im.save(filename)
    #     j = j + 1
    
    return finalpred
########################################################################
if __name__ == "__main__":   
    folder = './dataset/BasketballDrill_832x480_50/'
    folder_save = './dataset/BasketballDrill_832x480_50_residue12result/'

    coarse_json_dir = './models/BlowingBubbles_416x240_50_residue16_coarse.json'
    coarse_hdf5_dir = "./models/BlowingBubbles_416x240_50_residue16_coarse.hdf5"

    b1_json_dir = './models/BlowingBubbles_416x240_50_residue16_b1.json'
    b1_hdf5_dir = "./models/BlowingBubbles_416x240_50_residue16_b1.hdf5"

    b23_json_dir = './models/BlowingBubbles_416x240_50_residue16_b23.json'
    b23_hdf5_dir = "./models/BlowingBubbles_416x240_50_residue16_b23.hdf5"

    
    b = 16 # blk_size & ref. blk size
    test_start, test_end = 0, 240
    
    images =  load_imgs(folder, test_start, test_end)
    coarse_frames = coarse16_test(images, b)

    bm = 8 # target block size to predict
    N_frames = test_end - test_start
    
    decoded = regroup(N_frames, images.shape, b, coarse_frames)

    predicted_b1 = pred_inference_b1(N_frames, b, bm, images.shape, decoded)
    predicted_b1_frame = regroup(N_frames - 4, images.shape, bm, predicted_b1)

    predicted_b2 = pred_inference_b23(N_frames, b, bm, images.shape, decoded[0:N_frames - 4], predicted_b1_frame)
    predicted_b2_frame = regroup(N_frames - 4, images.shape, bm, predicted_b2)

    predicted_b3 = pred_inference_b23(N_frames, b, bm, images.shape, decoded[4:N_frames], predicted_b1_frame)
    predicted_b3_frame = regroup(N_frames - 4, images.shape, bm, predicted_b3)

    final_coarse = residue_inference(N_frames, images, decoded, folder_save, coarse_json_dir, coarse_hdf5_dir)
    final_predicted_b1 = residue_inference(N_frames-4, images[2:N_frames-2], predicted_b1_frame, folder_save, b1_json_dir, b1_hdf5_dir)
    final_predicted_b2 = residue_inference(N_frames-4, images[1:N_frames-3], predicted_b2_frame, folder_save, b23_json_dir, b23_hdf5_dir)
    final_predicted_b3 = residue_inference(N_frames-4, images[3:N_frames-1], predicted_b3_frame, folder_save, b23_json_dir, b23_hdf5_dir)
    
    # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    # print(final_coarse.shape)
    # print(final_predicted_b1.shape)
    # print(final_predicted_b2.shape)
    # print(final_predicted_b3.shape)

    amse, apsnr, assim = performance_evaluation(N_frames, images, final_coarse)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('average test final_amse:',amse)
    print('average test final_apsnr:',apsnr)
    print('average test final_assim:',assim)

    amse, apsnr, assim = performance_evaluation(N_frames-4, images[2:N_frames-2], final_predicted_b1)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('average test final_amse:',amse)
    print('average test final_apsnr:',apsnr)
    print('average test final_assim:',assim)

    amse, apsnr, assim = performance_evaluation(N_frames-4, images[1:N_frames-3], final_predicted_b2)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('average test final_amse:',amse)
    print('average test final_apsnr:',apsnr)
    print('average test final_assim:',assim)

    amse, apsnr, assim = performance_evaluation(N_frames-4, images[3:N_frames-1], final_predicted_b3)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('average test final_amse:',amse)
    print('average test final_apsnr:',apsnr)
    print('average test final_assim:',assim)

    


