import cv2
import math
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

def psnr(img_true, img_recovered):    
    pixel_max = 255.0
    mse = np.mean((img_true-img_recovered)**2)
    p = 20 * math.log10( pixel_max / math.sqrt( mse ))
    return p

def performance_evaluation(images, finalpred, start, step):
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
    j = start
    for result in finalpred:
        filename = save_dir + str(j) + '.png'
        im_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        im.save(filename)
        j = j + 1

# # Convert image frames to bxb blocks
# # (n_frame, width height, 3) -> (n_block, b*b, 3) -> (n_block, b, b, 3)
# def image_to_block(N_frames, images, b_size, skip):
#     coarse_set = []
#     # for img in images:
#     for i in range(0, N_frames): 
#         img = images[i + skip]
#         #b_size: blk_size
#         for y in range(0, img.shape[0], b_size):
#             for x in range(0, img.shape[1], b_size):
#                 block = img[y:y + b_size, x:x + b_size]
#                 block = block.reshape(b_size*b_size, 3)
#                 coarse_set.append(block)
    
#     coarse_set = np.array(coarse_set)
#     # (n_block, b*b, 3) -> (n_block, b, b, 3) # Daniel
#     coarse_set2 = coarse_set.reshape(coarse_set.shape[0], b_size, b_size, 3)

#     return coarse_set2

def image_padding(image, pad_size, mode='constant', constant_values=0):
        npad = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
        if mode == 'constant':
            return np.pad(image, npad, mode, constant_values=constant_values)
        else:
            return np.pad(image, npad, mode)
from skimage.util import view_as_blocks, view_as_windows
def image_to_block(images, b_size=8, pad_en=False, bm_size = 8):
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

# get input blocks to prediction model from decoded frames
# (100, 240, 416, 3) -> (149760, 16, 16, 3)
def get_block_set(N_frames, decoded, b, bm, step): 
    block_set = []

    for i in range(0, N_frames): 
        img = decoded[i + step]

        for y in range(0, img.shape[0], bm):
            for x in range(0, img.shape[1], bm):
                block = np.zeros((b, b, 3))              
                # bottom L
                if (y + bm/2+b/2) >= img.shape[0] and (x+bm/2 - b/2) < 0 :
                    block = img[y+bm-b:y+bm, x:x+b]
                    block = block.reshape(b*b,3)
                    block_set.append(block)               
                # bottom R
                elif (y + bm/2+b/2) >= img.shape[0] and (x+bm/2 + b/2) >= img.shape[1] :
                    block = img[y+bm-b:y+bm, x+bm-b:x+bm]
                    block = block.reshape(b*b,3)
                    block_set.append(block)
                # top L
                elif (y + bm/2-b/2) < 0 and (x+bm/2 - b/2) < 0 :
                    block = img[y:y+b, x:x+b]
                    block = block.reshape(b*b,3)
                    block_set.append(block)                    
                # top R
                elif (y + bm/2-b/2) < 0 and (x+bm/2 + b/2) >= img.shape[1] :
                    block = img[y:y+b, x+bm-b:x+bm]
                    block = block.reshape(b*b,3)
                    block_set.append(block)            
                # top row
                elif (y + bm/2-b/2) < 0 :
                    block = img[y:y+b, int(x+bm/2-b/2):int(x+bm/2+b/2)]
                    block = block.reshape(b*b,3)
                    block_set.append(block)                    
                # bottom row
                elif (y + bm/2+b/2) >= img.shape[0] :
                    block = img[y+bm-b:y+bm, int(x+bm/2-b/2):int(x+bm/2+b/2)]
                    block = block.reshape(b*b,3)
                    block_set.append(block)                    
                # left column
                elif (x+bm/2 - b/2) < 0 :
                    block = img[int(y+bm/2-b/2):int(y+bm/2+b/2), x:x+b]
                    block = block.reshape(b*b,3)
                    block_set.append(block)                    
                # right column
                elif (x+bm/2 + b/2) >= img.shape[1] :
                    block = img[int(y+bm/2-b/2):int(y+bm/2+b/2), x+bm-b:x+bm]
                    block = block.reshape(b*b,3)
                    block_set.append(block)                   
                else: # normal
                    block = img[int(y+bm/2-b/2):int(y+bm/2+b/2), int(x+bm/2-b/2):int(x+bm/2+b/2)]
                    block = block.reshape(b*b, 3)
                    block_set.append(block)

    block_set = np.array(block_set)
    block_set = block_set.reshape(block_set.shape[0], b, b, 3)

    return block_set