"""
Created by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 
"""
import cv2
import math
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from skimage.util import view_as_blocks, view_as_windows

def psnr(images, finalpred):    
    """
    Evaluate the PSNR of a predicted image
    :param images: a numpy array of a ground truth image with shape (hight, width, n_channel=3)
    :param finalpred: a numpy array of a final predicted truth image with shape (hight, width, n_channel=3)
    :paran: the PSNR of a predicted image
    """
    pixel_max = 255.0
    mse = np.mean((images-finalpred)**2)
    p = 20 * math.log10( pixel_max / math.sqrt( mse ))
    return p

def add_noise(e):
    """
    Add a noise layer between encoder and decoder
    param e: a tensor layer 
    return: a tensor layer added with uniform noise
    """
    shape = tuple(e.get_shape().as_list()[1:])
    noise = np.random.uniform(-0.5, 0.5, shape)
    noise_tensor = tf.constant(noise, e.dtype)
    e = e + noise_tensor
    return e

def performance_evaluation(images, finalpred, start, step):
    """
    Evaluate predicted images by MSE, PSNR, and SSIM
    :param images: a numpy array of float32 ground truth images with shape (n_image, hight, width, n_channel=3)
    :param finalpred: a numpy array of float32 final predicted truth images with shape (n_image, hight, width, n_channel=3)
    :param start: the startig index for evalutation. Must between [0, n_image - step)
    :param step: the number of images skipped in each iteration
    :return
        n_evaluated: the number of images being evaluated
        amse: average MSE
        psnr: average PSNR
        assim: average SSIM
    """
    finalpred = finalpred.astype('uint8')

    N_frames = images.shape[0]
    width, height = images.shape[1], images.shape[2]
    pixel_max = 255.0
    mse = []
    psnr = []
    ssims = []
   
    for i in range(start, N_frames, step):
        img = np.array(images[i].reshape(width*height, 3))
        res = finalpred[i].reshape(width*height, 3)
        m = mean_squared_error(img, res)
        s = ssim(img, res, multichannel=True)
        p = 20 * math.log10( pixel_max / math.sqrt( m ))
        psnr.append(p)
        mse.append(m)
        ssims.append(s)
    
    amse = np.mean(mse)
    apsnr = np.mean(psnr)
    assim = np.mean(ssims)
    n_evaluated = len(mse)
    return n_evaluated, amse, apsnr, assim

def load_imgs(path, start, end):
    """
    Load PNG images from path
    :param path: path where images are loaded from
    :param start: the starting index of image to be loaded
    :param end: the ending index of image to be loaded
    :return: a numpy array of float32 images with shape (n_image, hight, width, n_channel=3)
    """
    train_set = []
    for n in range(start, end):
        fname = path +  str(n) + ".png"
        img = cv2.imread(fname, 1)
        if img is not None:
                train_set.append(img)
    train_set = np.array(train_set)
    return train_set

# Save final frames
def save_imgs(save_dir, start, finalpred):
    """
    Save image as PNG format
    :param save_dir: the saving directry
    :param start: the starting index of image to be saved
    :finalpred: numpy array of float32 final predicted truth images with shape (n_image, hight, width, n_channel=3)
    """
    j = start
    for result in finalpred:
        filename = save_dir + str(j) + '.png'
        im_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        im.save(filename)
        j = j + 1

def image_padding(images, pad_size, mode='constant', constant_values=0):
    """
    Pad image frames with zero padding
    :param images: a numpy array of float32 images with shape (n_image, hight, width, n_channel=3)
    :param pad_size: the number of padding pixels added to each edge. 
    :return: padded images with shape (n_image, hight+2*pad_size, width+2*pad_size, n_channel=3) 
    """
    npad = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    if mode == 'constant':
        return np.pad(images, npad, mode, constant_values=constant_values)
    else:
        return np.pad(images, npad, mode)

def image_to_block(images, b_size=8, pad_en=False, bm_size = 8):
    """
    Patch Mode (pad_en=False)
        Slicing image frames to patches 
    Block Mode (pad_en=True)
        Slicing image frames to blocks that each block centers at a patch
    :param images: a numpy array of float32 images with shape (n_image, hight, width, n_channel=3)
    :param b_size: size of the output patch/block
    :param pad_en: padding switch
    :bm_size: centered patch size in Block Mode
    :return: 
        Patch Mode: (n_patche, b_size, b_size, 3)
        Block Mode: (n_block, b_size, b_size, 3)
    """
    blocks = []
    for img in images:
        if not pad_en:
            blocks.append(view_as_blocks(img, (b_size, b_size, 3)))
        else:
            padded_image = image_padding(img, (b_size-bm_size) >> 1)
            blocks.append(view_as_windows(padded_image, (b_size, b_size, 3), bm_size))
    return np.asarray(blocks).reshape((-1, b_size, b_size, 3))

# (n_blocks, b, b, 3) -> (n_frames, 480, 832, 3)
def regroup(N_frames, images_shape, predicted_frames, b_size):
    """
    Regroup patches/blocks to video frames
    :param N_frames: the number of frames grouped into 
    :param images_shape: a tuple of images shape (n_image, hight, width, n_channel=3)
    :param predicted_frames: a numpy array of image blocks/patches with shape (n_patch, b_size, b_size, , n_channel=3)
    :param b_size: size of the output patch/block
    :return: a numpy array of regrouped images frames with shape: (n_image, hight, width, n_channel=3)
    """
    width, height = images_shape[1], images_shape[2]
    
    final_prediction=[]
    
    i = 0
    for n in range(N_frames):
        result = np.zeros((width, height, 3))
        
        for y in range(0, width, b_size):
           for x in range(0, height, b_size):
               result[y:y + b_size, x:x + b_size,:] = predicted_frames[i].reshape(b_size,b_size,3)
               i = i + 1
              
        final_prediction.append(result)
    
    final_prediction = np.array(final_prediction) # re-group the decoded frames
        
    return final_prediction