"""
Created by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 
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

from utility.helper import psnr, load_imgs, regroup, performance_evaluation, save_imgs, image_to_block
import coarse.test
from utility.parameter import *
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

def residue_inference(images, pred, b, ratio, model, mode = 'noise'): 

    # =================================================
    json_path, hdf5_path = get_model_path(model, ratio)
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
    residue_model.compile(optimizer=opt, loss=keras.losses.MeanAbsoluteError())

    # ===================================================
    encoder_model = Model(inputs=residue_model.input,
                                          outputs=residue_model.get_layer('conv2d_3').output)

    residue = images - pred
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

    final2 = regroup(images.shape[0], images.shape, residue_frames, b)
    
    finalpred = np.add(pred, final2)
    
    return finalpred
########################################################################
def main(args = 1):       
    b = 16 
    bm = 8 
    
    test_images =  load_imgs(data_dir, test_start, test_end)
    decoded = coarse.test.predict(test_images, b, testing_ratio)

    predicted_b1_frame = pred_inference_b1(decoded, b, bm, testing_ratio)
    final_predicted_b1 = residue_inference(test_images[2:-2], predicted_b1_frame, b, testing_ratio, "residue_b1")

    predicted_b2_frame = pred_inference_b23(decoded[:-4], final_predicted_b1, b, bm, testing_ratio)
    final_predicted_b2 = residue_inference(test_images[1:-3], predicted_b2_frame, b, testing_ratio, "residue_b23")

    predicted_b3_frame = pred_inference_b23(decoded[4:], final_predicted_b1, b, bm, testing_ratio)
    final_predicted_b3 = residue_inference(test_images[3:-1], predicted_b3_frame, b, testing_ratio, "residue_b23")

# #################################### EVALUATION #####################################

    n_predicted1, amse1, apsnr1, assim1 \
        = performance_evaluation(test_images[2:-2], final_predicted_b1, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_b1:',n_predicted1)
    print('average test b1_amse:',amse1)
    print('average test b1_apsnr:',apsnr1)
    print('average test b1_assim:',assim1)

    n_predicted2, amse2, apsnr2, assim2 \
        = performance_evaluation(test_images[1:-3], final_predicted_b2, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_b2:',n_predicted2)
    print('average test b2_amse:',amse2)
    print('average test b2_apsnr:',apsnr2)
    print('average test b2_assim:',assim2)

    n_predicted3, amse3, apsnr3, assim3 \
        = performance_evaluation(test_images[3:-1], final_predicted_b3, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_b3:',n_predicted3)
    print('average test b3_amse:',amse3)
    print('average test b3_apsnr:',apsnr3)
    print('average test b3_assim:',assim3)
    
if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])