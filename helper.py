import cv2

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