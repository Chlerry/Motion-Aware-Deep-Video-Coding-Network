import cv2
import numpy as np

def psnr(img_true, img_recovered):    
    pixel_max = 255.0
    mse = np.mean((img_true-img_recovered)**2)
    p = 20 * math.log10( pixel_max / math.sqrt( mse ))
    return p

def load_imgs(path, start, end):
    train_set = []
    for n in range(start, end):
        fname = path +  str(n) + ".png"
        img = cv2.imread(fname, 1)
        if img is not None:
                train_set.append(img)
    train_set = np.array(train_set)
    return train_set

def get_coarse_set(images, b):
    coarse_set = []
    for img in images:
        #b: blk_size
        for y in range(0, img.shape[0], b):
            for x in range(0, img.shape[1], b):
                block = img[y:y + b, x:x + b]
                block = block.reshape(b*b, 3)
                coarse_set.append(block)
    
    coarse_set = np.array(coarse_set)
    # (n_block, b*b, 3) -> (n_block, b, b, 3) # Daniel
    coarse_set2 = coarse_set.reshape(coarse_set.shape[0], b, b, 3)

    return coarse_set2