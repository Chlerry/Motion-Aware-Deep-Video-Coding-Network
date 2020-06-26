"""
Refactored and updated by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 

Initially created by Ying Liu on Wed Feb 12 23:31:07 2020
"""
# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import keras
import numpy as np
from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from utility.parameter import *
from utility.helper import psnr, load_imgs, regroup, image_to_block, performance_evaluation
import coarse.test
import prediction.inference

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
    
def predict(residue, b, ratio, mode = 'default'): # start

    C = image_to_block(residue, b)
    
    # =================================================
    json_path, hdf5_path = get_model_path("residue", ratio)
    # =================================================
    # Load residue model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    residue_model = model_from_json(loaded_model_json, custom_objects={'tf': tf})

    # Load weights into new model
    residue_model.load_weights(hdf5_path)
    print("Loaded model from " + hdf5_path)

    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    residue_model.compile(optimizer=opt, loss='mse')
    
    # ===================================================
    encoder_model = Model(inputs=residue_model.input,
                                          outputs=residue_model.get_layer('conv2d_3').output)

    residue_set = image_to_block(residue, b)
    encoded = encoder_model.predict(residue_set) 

    encoded = np.rint(encoded)
    # ===================================================
    idx = 5 # index of desired layer
    layer_input = Input(shape=encoded.shape[1:]) # a new input tensor to be able to feed the desired layer
    
    # Create the new nodes for each layer in the path
    x = layer_input
    for layer in residue_model.layers[idx:]:
        x = layer(x)
     
    # Create the model
    decoder_model = Model(layer_input, x)

    residue_frames = decoder_model.predict(encoded)
    # ===================================================
    residue_frames = np.array(residue_frames)
    final_prediction = regroup(residue.shape[0], residue.shape, residue_frames, b)
    
    return final_prediction
  
def main(args = 1):   
    b = 16 
    bm = 8 

    test_images = load_imgs(data_dir, test_start, test_end)

    decoded = coarse.test.predict(test_images, b, testing_ratio)

    regrouped_prediction = prediction.inference.predict(decoded, b, bm, testing_ratio)

    residue_test_images = test_images[1:n_test_frames-1]
    residue = residue_test_images - regrouped_prediction

    residue_predicted_frames = predict(residue, b, testing_ratio)
    final_frame = np.add(regrouped_prediction, residue_predicted_frames)

    start, step = 0, 1

    n_evaluated, pred_res_amse, pred_res_apsnr, pred_res_assim\
         = performance_evaluation(residue_test_images, final_frame, start, step)
    print('n_evaluated:',n_evaluated)
    print('average test pred_res_amse:',pred_res_amse)
    print('average test pred_res_apsnr:',pred_res_apsnr)
    print('average test pred_res_assim:',pred_res_assim)

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])