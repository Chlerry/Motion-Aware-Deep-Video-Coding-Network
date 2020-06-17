# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

from utility.parameter import *
from utility.helper import psnr, load_imgs, regroup, save_imgs, performance_evaluation
import coarse.test 

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

def pred_inference_b1(decoded, b, bm, ratio):
    
    N_frames = decoded.shape[0]
    # ============== DL ===============================
    prev = image_to_block(decoded[:-4], b, True)

    B = image_to_block(decoded[4:], b, True)
    # ============== DL ===============================
    json_path, hdf5_path = get_model_path("prediction", ratio)

    # ============== YL: load model ===================
    
    from keras.models import model_from_json
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pred_model = model_from_json(loaded_model_json)
    # load weights into new model
    pred_model.load_weights(hdf5_path)
    print("Loaded model from " + hdf5_path)
    
    # ============== DL ===============================
    # evaluate loaded model on test data
    opt = tf.keras.optimizers.Adam()
    if rtx_optimizer == True:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    pred_model.compile(optimizer=opt, loss='mse', metrics=['acc'])
    # ===================================================

    predicted_frames = pred_model.predict([prev, B])

    predicted_b1_frame = regroup(N_frames - 4, decoded.shape, predicted_frames, bm)
    return predicted_b1_frame

def main(args = 1): 
    b = 16 # blk_size & ref. blk size
    bm = 8 # target block size to predict
    
    test_images = load_imgs(data_dir, test_start, test_end)
    decoded = coarse.test.predict(test_images, b, training_ratio)

    predicted_b1_frame = pred_inference_b1(decoded, b, bm, training_ratio)
    
    n_predicted1, amse1, apsnr1, assim1 \
        = performance_evaluation(test_images[2:n_test_frames-2], predicted_b1_frame, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_b1:',n_predicted1)
    print('average test b1_amse:',amse1)
    print('average test b1_apsnr:',apsnr1)
    print('average test b1_assim:',assim1)

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])