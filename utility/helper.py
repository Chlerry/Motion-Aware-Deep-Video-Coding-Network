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

def regroup(N_frames, images_shape, bm, predicted_frames):
    
    width, height = images_shape[1], images_shape[2]
    
    final_prediction=[]
    
    i = 0
    for n in range(N_frames):
        result = np.zeros((width, height, 3))
        
        for y in range(0, width, bm):
           for x in range(0, height, bm):
               res = x
               result[y:y + bm, x:x + bm,:] = predicted_frames[i].reshape(bm,bm,3)
               i = i + 1
              
        final_prediction.append(result)
    
    final_prediction = np.array(final_prediction) # re-group the decoded frames
        
    return final_prediction

def get_block_set(N_frames, decoded, b, bm, skip): 
    block_set = []

    for i in range(0, N_frames): 
        img = decoded[i + skip]

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