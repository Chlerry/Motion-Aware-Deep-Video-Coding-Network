<<<<<<< HEAD
<<<<<<< HEAD
=======
import cv2
import math
=======
# =================================================
>>>>>>> 91226fc... Re-format comments
# import keras
# from keras.models import Model
# from keras.layers import Input, Conv2D, Conv2DTranspose
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# import matplotlib.pyplot as plt

# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import EarlyStopping

# from skimage.measure import compare_ssim as ssim

# SEED=42
# =================================================
import cv2
import math
import numpy as np

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from PIL import Image

from helper import psnr, load_imgs, get_coarse_set

def results(coarse_frames, images, foldername,  test_start, test_end, b):
    
    print("RESULTS")
    width,height = images.shape[1],images.shape[2]
    N_mblocks = (width*height)/(b*b)
    N_test = test_end-test_start
    final = []
    j = test_start
    f = coarse_frames.reshape(-1, int(N_mblocks), int(b*b), 3)
    for n in f: # loop over test frames
        result = np.zeros((width, height, 3))
        i = 0
        for y in range(0, result.shape[0], b):
           for x in range(0, result.shape[1], b):
               res = n[i].reshape(b,b,3)
               result[y:y + b, x:x + b] = res
               i = i + 1
        filename = foldername +str(j)+'.png'
        im_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        im.save(filename)
        j = j + 1
        final.append(result)
    
    final = np.array(final)  
    
    pixel_max = 255.0
    mse = []
    psnr = []
    ssims = []
    
    for i in range(N_test): 
        img = np.array(images[i].reshape(width*height, 3), dtype = float)
        res = final[i].reshape(width*height, 3)

        m = mean_squared_error(img, res)
        p = 20 * math.log10( pixel_max / math.sqrt( m ))
        s = ssim(img, res, multichannel=True)
        psnr.append(p)
        ssims.append(s)
        mse.append(m)
    
    amse = np.mean(mse)
    apsnr = np.mean(psnr)
    assim = np.mean(ssims)
    return amse, apsnr, assim
    
    
def coarse16_test(test_start,test_end,folder, b, folder_save):
    
    images =  load_imgs(folder, test_start,test_end)
    
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
    
    # coarse_set = []
    # for img in images:
    #     for y in range(0, img.shape[0], b):
    #         for x in range(0, img.shape[1], b):
    #             block = img[y:y + b, x:x + b]
    #             block = block.reshape(b*b, 3)
    #             coarse_set.append(block)
    
    # coarse_set = np.array(coarse_set)
    # coarse_set2 = coarse_set.reshape(coarse_set.shape[0], b, b, 3)

    coarse_set = get_coarse_set(images, b)
    
    coarse_frames = coarse_model.predict(coarse_set)

    #foldername = '/home/yingliu/Desktop/Rida/Code/BlowingBubbles_416x240_50_coarse16result/'
    amse, apsnr, assim = results(coarse_frames, images, folder_save, test_start, test_end, b)
    return amse, apsnr, assim

if __name__ == "__main__":   
    folder = './dataset/BasketballDrill_832x480_50/'
    
    folder_save = 'coarse16result/BasketballDrill_832x480_50/'
    
    b = 16 # blk_size
    test_start, test_end = 0, 100
    #coarse16_train(train_start,train_end)
    amse, apsnr, assim = coarse16_test(test_start,test_end,folder, b, folder_save)
    print('average test mse:',amse)
    print('average test psnr:',apsnr)
    print('average test ssim:',assim)
    





>>>>>>> 5a942be... Replace psnr and load_imgs method with existing module
