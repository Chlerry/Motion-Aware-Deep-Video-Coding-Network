# Disable INFO and WARNING messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

from utility.parameter import *
from utility.helper import psnr, load_imgs, get_block_set, regroup, performance_evaluation
import coarse.test
from prediction.b1_inference import pred_inference_b1

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

def pred_inference_b23(prev_decoded, predicted_b1_frame, b, bm, ratio):
    
    N_frames = prev_decoded.shape[0]
    # ============== DL ===============================
    prev = get_block_set(N_frames, prev_decoded, b, bm, 0)
    # print(prev.shape)
    
    B = get_block_set(N_frames, predicted_b1_frame, b, bm, 0)
    # print(B.shape)
    # ============== DL ===============================
    json_path, hdf5_path = get_model_path("prediction_b23", ratio)
    
    # ============== YL: load model ===============================
    
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

    predicted_b23 = pred_model.predict([prev, B])

    predicted_b23_frame = regroup(n_test_frames - 4, prev_decoded.shape, predicted_b23, bm)
    return predicted_b23_frame
    # ===================================================
    
def main(args = 1): 
    
    b = 16 # blk_size & ref. blk size
    bm = 8 # target block size to predict
    
    test_images =  load_imgs(data_dir, test_start, test_end)
    decoded = coarse.test.predict(test_images, b, testing_ratio)

    predicted_b1_frame = pred_inference_b1(decoded, b, bm, testing_ratio)
    from residue.b_inference import residue_inference
    final_predicted_b1 = residue_inference(test_images[2:n_test_frames-2], predicted_b1_frame, b, "residue_b1", testing_ratio)

    predicted_b2_frame = pred_inference_b23(decoded[0:n_test_frames - 4], final_predicted_b1, b, bm, testing_ratio)

    n_predicted2, amse2, apsnr2, assim2 \
        = performance_evaluation(test_images[1:n_test_frames-3], predicted_b2_frame, 0, 4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('n_b2:',n_predicted2)
    print('average test b2_amse:',amse2)
    print('average test b2_apsnr:',apsnr2)
    print('average test b2_assim:',assim2)

if __name__ == "__main__":   
    import sys
    main(sys.argv[1:])