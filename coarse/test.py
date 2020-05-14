# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
from keras.models import model_from_json

from utility.parameter import *
from utility.helper import load_imgs, image_to_block, save_imgs, regroup, performance_evaluation

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
import keras.backend as K
if rtx_optimizer == True:
    K.set_epsilon(1e-4) 
# =================================================
    
def predict(images, b, ratio):
    # ============== DL ===============================
    json_path, hdf5_path = get_model_path("coarse", ratio)
    # =================================================

    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    coarse_model = model_from_json(loaded_model_json)

    # load weights into new model
    coarse_model.load_weights(hdf5_path)
    print("Loaded model from " + hdf5_path)
    
    # evaluate loaded model on test data
    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    coarse_model.compile(optimizer=opt, loss='mse', metrics=['acc'])

    coarse_set = image_to_block(images.shape[0], images, b, 0)
    coarse_frames = coarse_model.predict(coarse_set)

    coarse_frames = np.array(coarse_frames)
    decoded = regroup(images.shape[0], images.shape, coarse_frames, b)

    return decoded

def main(args = 1): 
    b = 16 # blk_size

    test_images =  load_imgs(data_dir, test_start,test_end)
    decoded = predict(test_images, b, testing_ratio)

    start, step = 0, 1
    n2_evaluated, coarse_amse, coarse_apsnr, coarse_assim \
        = performance_evaluation(test_images, decoded, start, step)
    print('n_evaluated:',n2_evaluated)
    print('average test coarse_amse:',coarse_amse)
    print('average test coarse_apsnr:',coarse_apsnr)
    print('average test coarse_assim:',coarse_assim)

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])