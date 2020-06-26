"""
Refactored and updated by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 

Initially created by Ying Liu on Wed Feb 12 23:31:07 2020
"""
# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import keras
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from utility.parameter import *
from utility.helper import psnr, load_imgs, image_to_block, performance_evaluation, regroup

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

def predict(decoded, b, bm, ratio, model = "prediction"):
    
    N_frames = decoded.shape[0]
    # =================================================
    prev = image_to_block(decoded[:-2], b, True)

    B = image_to_block(decoded[2:], b, True)
    # =================================================
    json_path, hdf5_path = get_model_path(model, ratio)
    
    from keras.models import model_from_json
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pred_model = model_from_json(loaded_model_json)
    # Load weights into new model
    pred_model.load_weights(hdf5_path)
    print("Loaded model from " + hdf5_path)
    
    # Evaluate loaded model on test data
    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    pred_model.compile(optimizer=opt, loss=keras.losses.MeanAbsoluteError(), metrics=['acc'])

    predicted_frames = pred_model.predict([prev, B])

    regrouped_prediction = regroup(N_frames - 2, decoded.shape, predicted_frames, bm)
    return regrouped_prediction
    # ===================================================
    
def main(args = 1):   
    
    b = 16 
    bm = 8 
    
    test_images = load_imgs(data_dir, test_start, test_end)
    import coarse.test
    decoded = coarse.test.predict(test_images, b, testing_ratio)

    regrouped_prediction = predict(decoded, b, bm, testing_ratio)

    start, step = 0, 1
    n0_evaluated, pred_amse, pred_apsnr, pred_assim \
        = performance_evaluation(test_images[1:n_test_frames-1], regrouped_prediction, start, step)
    print('n0_evaluated:',n0_evaluated)
    print('average test pred_amse:',pred_amse)
    print('average test pred_apsnr:',pred_apsnr)
    print('average test pred_assim:',pred_assim)

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])