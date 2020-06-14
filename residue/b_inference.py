# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from utility.helper import psnr, load_imgs, regroup, performance_evaluation, save_imgs, image_to_block
import coarse.test
from utility.parameter import *
from prediction.b1_inference import pred_inference_b1
from prediction.b23_inference import pred_inference_b23

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

def residue_inference(images, pred, b, model, ratio): # start
    N_frames = images.shape[0]

    residue = images - pred
    
    C = image_to_block(residue, b)

    # ============== DL ===============================
    json_path, hdf5_path = get_model_path(model, ratio)
    # =================================================
    # load residue16 model
    from keras.models import model_from_json

    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    residue_model = model_from_json(loaded_model_json)
    # load weights into new model

    residue_model.load_weights(hdf5_path)
    print("Loaded model from " + hdf5_path)
    
    # ============== DL ===============================
    # evaluate loaded model on test data
    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    residue_model.compile(optimizer=opt, loss='mse')
    # ===================================================
    residue_decoded = residue_model.predict([C])
    
    final2 = regroup(N_frames, images.shape, residue_decoded, b)
    
    finalpred = np.add(pred, final2)
    
    return finalpred
########################################################################
def main(args = 1):       
    b = 16 # blk_size & ref. blk size
    bm = 8 # target block size to predict
    
    test_images =  load_imgs(data_dir, test_start, test_end)
    decoded = coarse.test.predict(test_images, b, testing_ratio)

    predicted_b1_frame = pred_inference_b1(decoded, b, bm, testing_ratio)
    final_predicted_b1 = residue_inference(test_images[2:n_test_frames-2], predicted_b1_frame, b, "residue_b1", testing_ratio)

    predicted_b2_frame = pred_inference_b23(decoded[0:n_test_frames - 4], final_predicted_b1, b, bm, testing_ratio)
    final_predicted_b2 = residue_inference(test_images[1:n_test_frames-3], predicted_b2_frame, b, "residue_b23", testing_ratio)

    predicted_b3_frame = pred_inference_b23(decoded[4:n_test_frames], final_predicted_b1, b, bm, testing_ratio)
    final_predicted_b3 = residue_inference(test_images[3:n_test_frames-1], predicted_b3_frame, b, "residue_b23", testing_ratio)

#################################### EVALUATION #####################################
    n_predicted0, amse0, apsnr0, assim0 \
        = performance_evaluation(test_images, decoded, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_decoded:',n_predicted0)
    print('average test coarse_amse:',amse0)
    print('average test coarse_apsnr:',apsnr0)
    print('average test coarse_assim:',assim0)

    n_predicted1, amse1, apsnr1, assim1 \
        = performance_evaluation(test_images[2:n_test_frames-2], final_predicted_b1, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_b1:',n_predicted1)
    print('average test b1_amse:',amse1)
    print('average test b1_apsnr:',apsnr1)
    print('average test b1_assim:',assim1)

    n_predicted2, amse2, apsnr2, assim2 \
        = performance_evaluation(test_images[1:n_test_frames-3], final_predicted_b2, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_b2:',n_predicted2)
    print('average test b2_amse:',amse2)
    print('average test b2_apsnr:',apsnr2)
    print('average test b2_assim:',assim2)

    n_predicted3, amse3, apsnr3, assim3 \
        = performance_evaluation(test_images[3:n_test_frames-1], final_predicted_b3, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_b3:',n_predicted3)
    print('average test b3_amse:',amse3)
    print('average test b3_apsnr:',apsnr3)
    print('average test b3_assim:',assim3)

    n_predicted = n_predicted0 + n_predicted1 + n_predicted2 + n_predicted3
    amse = (amse0*n_predicted0 + amse1*n_predicted1 + amse2*n_predicted2 + amse3*n_predicted3) / n_predicted
    apsnr = (apsnr0*n_predicted0 + apsnr1*n_predicted1 + apsnr2*n_predicted2 + apsnr3*n_predicted3) / n_predicted
    assim = (assim0*n_predicted0 + assim1*n_predicted1 + assim2*n_predicted2 + assim3*n_predicted3) / n_predicted

    print("vvvvvvvvvvvv Overall vvvvvvvvvvvvv")
    print('average test final_amse:',amse)
    print('average test final_apsnr:',apsnr)
    print('average test final_assim:',assim)
    
if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])