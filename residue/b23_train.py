"""
Created by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 
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
import coarse.test
from prediction.b1_inference import pred_inference_b1
from prediction.b23_inference import pred_inference_b23

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
      
from residue.b1_train import residue_train
    
def main(args = 1): 
    b = 16 
    bm = 8 
    
    train_images =  load_imgs(data_dir, train_start, train_end)
    decoded = coarse.test.predict(train_images, b, training_ratio)

    predicted_b1_frame = pred_inference_b1(decoded, b, bm, training_ratio,"prediction_b1")
    from residue.b_inference import residue_inference
    final_predicted_b1 = residue_inference( \
        train_images[2:-2], predicted_b1_frame, b, training_ratio, "residue_b1")

    predicted_b2_frame = pred_inference_b23( \
        decoded[:-4], final_predicted_b1, b, bm, training_ratio)

    residue = train_images[1:-3] - predicted_b2_frame
    residue_train(residue, b, training_ratio, "residue_b23")
    
if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])