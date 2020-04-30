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

from utility.test_result import coarse_results
from utility.helper import psnr, load_imgs, get_coarse_set

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
    
def coarse16_test(images, b):
    
    from keras.models import model_from_json
    json_file = open('./models/BlowingBubbles_416x240_50_coarse12.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    coarse_model = model_from_json(loaded_model_json)
    # load weights into new model
    coarse_model.load_weights("./models/BlowingBubbles_416x240_50_coarse12.hdf5")
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    coarse_model.compile(optimizer='adam', loss='mse', metrics=['acc'])

    coarse_set = get_coarse_set(images, b)
    
    coarse_frames = coarse_model.predict(coarse_set)
    
    return coarse_frames

if __name__ == "__main__":   
    folder = './dataset/BasketballDrill_832x480_50/'
    
    folder_save = 'coarse16result/BasketballDrill_832x480_50/'
    
    b = 16 # blk_size
    test_start, test_end = 0, 100
    #coarse16_train(train_start,train_end)

    images =  load_imgs(folder, test_start,test_end)
    coarse_frames = coarse16_test(images, b)

    amse, apsnr, assim = coarse_results(coarse_frames, images, folder_save, test_start, test_end, b)

    print('average test mse:',amse)
    print('average test psnr:',apsnr)
    print('average test ssim:',assim)
    
