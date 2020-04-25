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




def psnr(img_true, img_recovered):    
    pixel_max = 255.0
    mse = np.mean((img_true-img_recovered)**2)
    p = 20 * math.log10( pixel_max / math.sqrt( mse ))
    return p


def load_imgs(path, start, end):
    train_set = []
    for n in range(start, end):
        fname = path  + str(n) + ".png"
        img = cv2.imread(fname, 1)
        if img is not None:
                train_set.append(img)
    train_set = np.array(train_set)
    return train_set

def coarse16_inference(folder, start, end):
    images =  load_imgs(folder, start, end)
    
    print(images.shape)
    
    #images =  load_imgs(folder, test_start,test_end)

    
    
    from keras.models import model_from_json
    json_file = open('./models/BlowingBubbles_416x240_50_coarse16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    coarse_model = model_from_json(loaded_model_json)
    # load weights into new model
    coarse_model.load_weights("./models/BlowingBubbles_416x240_50_coarse16.hdf5")
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    coarse_model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    
    
    coarse_set = []
    for img in images:
        block = []
        for y in range(0, img.shape[0], b):
            for x in range(0, img.shape[1], b):
                block = img[y:y + b, x:x + b]
                block = block.reshape(b*b, 3)
                coarse_set.append(block)
    
    coarse_set = np.array(coarse_set)
    coarse_set2 = coarse_set.reshape(coarse_set.shape[0], b, b, 3)
    
    coarse_frames = coarse_model.predict(coarse_set2)

    return coarse_frames

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
    json_file = open('./models_new/BlowingBubbles_416x240_50_pred16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pred_model = model_from_json(loaded_model_json)
    # load weights into new model
    pred_model.load_weights("./models_new/BlowingBubbles_416x240_50_pred16.hdf5")
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
    json_file = open('./models_new/BlowingBubbles_416x240_50_residue16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    residue_model = model_from_json(loaded_model_json)
    # load weights into new model
    residue_model.load_weights("./models_new/BlowingBubbles_416x240_50_residue16.hdf5")
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
    
    coarse_frames = coarse16_inference(folder, test_start, test_end)
    bm = 8 # target block size to predict
    predicted_frames = pred_inference(folder, test_start, test_end, b, bm, coarse_frames)
    pred_amse, pred_apsnr, pred_assim = performance_evaluation(folder,test_start,test_end,predicted_frames)

    final_frames = residue_inference(folder, test_start, test_end, predicted_frames, folder_save)
    final_amse, final_apsnr, final_assim = performance_evaluation(folder,test_start,test_end,final_frames)

