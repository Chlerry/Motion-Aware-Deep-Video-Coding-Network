import helper.psnr as psnr
import helper.load_imgs as load_imgs

if __name__ == "__main__":   
    folder = './dataset/BlowingBubbles_416x240_50/'
    b = 16 # blk_size
    train_start, train_end = 0, 100
    coarse16_train(train_start,train_end, folder)