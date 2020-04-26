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

from helper import psnr, load_imgs
from coarse_test import coarse16_test
from prediction_inference import pred_inference
from residue_train import regroup

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

def performance_evaluation(folder,start,end,finalpred):
    images =  load_imgs(folder, start+1, end-1)
    N_frames = end-start
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
    
def residue_inference(folder, start, end, pred, folder_save): # start
    N_frames = end-start

    images = load_imgs(folder, start+1, end-1)
    width, height = images.shape[1], images.shape[2]
    
    #N_blocks = int((width*height)/(b*b))
    

    
    residue = images - pred
    print(residue.shape)
    
    
    # =================== reshape original images ===============
    C_ori = []
    
    for i in range(0, N_frames-2): 
        current = images[i]
        for y in range(0, width, b):
            for x in range(0, height, b):
                block = current[y:y + b, x:x + b]
                block = block.reshape(b*b, 3)
                C_ori.append(block)
    
    C_ori = np.array(C_ori)
    C_ori = C_ori.reshape(C_ori.shape[0], b, b, 3)
    print(C_ori.shape) # original blocks
    # ======================================================
    
    # =================== reshape predicted images ===============
    C_pred = []
    
    for i in range(0, N_frames-2): 
        current = pred[i]
        for y in range(0, width, b):
            for x in range(0, height, b):
                block = current[y:y + b, x:x + b]
                block = block.reshape(b*b, 3)
                C_pred.append(block)
    
    C_pred = np.array(C_pred)
    C_pred = C_pred.reshape(C_pred.shape[0], b, b, 3)
    print(C_pred.shape) # predicted blocks
    # ======================================================
    
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
    
# load residue16 model
    from keras.models import model_from_json
    json_file = open('./models/BlowingBubbles_416x240_50_residue16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    residue_model = model_from_json(loaded_model_json)
    # load weights into new model
    residue_model.load_weights("./models/BlowingBubbles_416x240_50_residue16.hdf5")
    print("Loaded model from disk")
    
    residue_decoded = residue_model.predict([C])
    
    final2 = []
    N_mblocks = int((width*height)/(b*b))
    f = residue_decoded.reshape(N_frames-2, N_mblocks, b*b, 3)
    for n in f:
        result = np.zeros((width, height,3))
        i = 0
        for y in range(0, result.shape[0], b):
            for x in range(0, result.shape[1], b):
                res = n[i].reshape(b,b,3)
                result[y:y + b, x:x + b] = res
                i = i + 1
        final2.append(result)
    
    final2 = np.array(final2)    
    
    finalpred = np.add(pred, final2)
  
    
    j = start+1
    for result in finalpred:
        filename = folder_save+str(j)+'.png'
        im_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        im.save(filename)
        j = j + 1
    
    return finalpred
  
if __name__ == "__main__":   
    folder = './dataset/BlowingBubbles_416x240_50/'
    
    folder_save = './dataset/BlowingBubbles_416x240_50_residue16result/'
    
    b = 16 # blk_size & ref. blk size
    test_start, test_end = 100, 120
    
    images =  load_imgs(folder, test_start, test_end)
    coarse_frames = coarse16_test(images, b)

    bm = 8 # target block size to predict
    N_frames = test_end - test_start

    predicted_frames = pred_inference(N_frames, b, bm, images, coarse_frames)
    final_prediction = regroup(N_frames, images.shape, bm, predicted_frames)
    pred_amse, pred_apsnr, pred_assim = performance_evaluation(folder,test_start,test_end,final_prediction)

    final_frames = residue_inference(folder, test_start, test_end, final_prediction, folder_save)
    final_amse, final_apsnr, final_assim = performance_evaluation(folder,test_start,test_end,final_frames)
