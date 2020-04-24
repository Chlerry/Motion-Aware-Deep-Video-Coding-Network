import helper.psnr as psnr
import helper.load_imgs as load_imgs

def coarse16_train(f_start,f_end,folder):
    images =  load_imgs(folder, f_start, f_end)  

if __name__ == "__main__":   
    folder = './dataset/BlowingBubbles_416x240_50/'
    b = 16 # blk_size
    train_start, train_end = 0, 100
    coarse16_train(train_start,train_end, folder)