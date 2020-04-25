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

def pred_inference(folder, start, end, b, bm, coarse_frames):
    
    N_frames = end - start
        
    #coarse_frames = coarse16_pred(folder, start, end)
    images = load_imgs(folder, start, end)
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
    #final_prediction = np.zeros((N_frames-2,width,height,3))
    final_prediction=[]
    
    #f = coarse_frames.reshape(N_frames, N_blocks, b*b, 3)
    i=0
    for n in range(N_frames-2):
        result = np.zeros((width, height, 3))
        
        for y in range(0, width, bm):
           for x in range(0, height, bm):
               res = x
               result[y:y + bm, x:x + bm,:] = predicted_frames[i].reshape(bm,bm,3)
               i = i + 1
              
        final_prediction.append(result)
    
    final_prediction = np.array(final_prediction) # re-group the decoded frames
        
    return final_prediction
    # ===================================================

def performance_evaluation(folder,start,end,finalpred):
    images =  load_imgs(folder, start+1, end-1)
    N_frames = end-start
    width, height = images.shape[1], images.shape[2]
    pixel_max = 255.0
    mse = []
    psnr = []
    ssims = []
    
   
    for i in range(0,N_frames-2):
        img = images[i].reshape(width*height, 3)
        res = finalpred[i].reshape(width*height, 3)
        m = mean_squared_error(img, res)
        s = ssim(img, res, multichannel=True)
        p = 20 * math.log10( pixel_max / math.sqrt( m ))
        psnr.append(p)
        mse.append(m)
        ssims.append(s)
    
    j = start+1
    for result in finalpred:
        filename = 'prediction16result/'+str(j)+'.png'
        im_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        im.save(filename)
        j = j + 1
    
    amse = np.mean(mse)
    apsnr = np.mean(psnr)
    assim = np.mean(ssims)
    return amse, apsnr, assim
      
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

    
def residue_inference(folder, start, end, pred): # start
    N_frames = end-start

    images = load_imgs(folder, start+1, end-1)
    width, height = images.shape[1], images.shape[2]
    
    #N_blocks = int((width*height)/(b*b))
    
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
    
# load residue16 model
    from keras.models import model_from_json
    json_file = open('./models/BlowingBubbles_416x240_50_residue16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    residue_model = model_from_json(loaded_model_json)
    # load weights into new model
    residue_model.load_weights("./models/BlowingBubbles_416x240_50_residue16.hdf5")
    print("Loaded model from disk")
    
    residue_decoded = residue_model.predict(C)
    
    final2 = []
    N_mblocks = (width*height)/(b*b)
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
        filename = 'residue16/'+str(j)+'.png'
        im_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        im.save(filename)
        j = j + 1
    
if __name__ == "__main__":   
    folder = './dataset/BlowingBubbles_416x240_50/'
    
    b = 16 # blk_size & ref. blk size
    test_start, test_end = 100, 200
    train_start, train_end = 0, 20

    images =  load_imgs(train_start, train_end, end)
    coarse_frames = coarse16_test(images, b)
    bm = 8 # target block size to predict
    predicted_frames = pred_inference(folder, train_start, train_end, b, bm, coarse_frames)
 
    residue_train(folder, train_start, train_end, bm, b, predicted_frames)
    