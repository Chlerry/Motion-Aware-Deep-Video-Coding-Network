from helper import psnr as psnr
from helper import load_imgs 

import numpy as np
from keras.layers import * # YL
from keras.models import Model


from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

def coarse16_train(f_start,f_end,folder):
    images =  load_imgs(folder, f_start, f_end)  

    coarse_train_set = []
    for img in images:
        block = []

        #b: blk_size
        for y in range(0, img.shape[0], b):
            for x in range(0, img.shape[1], b):
                block = img[y:y + b, x:x + b]
                block = block.reshape(b*b, 3)
                coarse_train_set.append(block)
    
    coarse_train_set = np.array(coarse_train_set)
    # (n_block, b*b, 3) -> (n_block, b, b, 3)
    coarse_train_set2 = coarse_train_set.reshape(coarse_train_set.shape[0], b, b, 3)
    
    input_coarse = Input(shape = (b, b, 3))
    
    e = Conv2D(128, kernel_size=(5, 5), padding = "SAME", strides = (4,4), activation='relu', input_shape=(b, b, 3))(input_coarse)
    print(e.shape)
    #e = MaxPool2D(strides=(2,2))(e)
    # stride (1,1)
    e = Conv2D(64, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    print(e.shape)
    #e = MaxPool2D(strides=(2,2))(e)
    e = Conv2D(3, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    
    d = Conv2DTranspose(64, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(e)
    d = Conv2DTranspose(128, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu')(d)
    d = Conv2DTranspose(3, kernel_size=(5, 5), padding = "SAME", strides = (4, 4), activation='relu')(d)
    
    coarse_model = Model(inputs = input_coarse, outputs = d)
    coarse_model.summary()
    coarse_model.compile(optimizer='adam', loss='mse') # RK
    # from keras.optimizers import Adam #YL
    # from sklearn.metrics import mean_squared_error
    # coarse_model.compile(optimizer=Adam(2e-4), loss='mean_squared_error', metrics=['acc']) # YL
    
    # ============== YL ===============================
    # save model and load model
    # from keras.models import model_from_json
    # serialize model to JSON
    model_json = coarse_model.to_json()
    with open("./models/BlowingBubbles_416x240_50_coarse16.json", "w") as json_file:
        json_file.write(model_json)

    
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=2, patience=10, \
                              verbose=2, mode='auto', \
                              baseline=None, restore_best_weights=True)                    
    # define modelcheckpoint callback
    checkpointer = ModelCheckpoint(filepath='./models/BlowingBubbles_416x240_50_coarse16.hdf5',\
                                   monitor='val_loss',save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    coarse_model.fit(coarse_train_set2, coarse_train_set2, batch_size=10, epochs=1, verbose=2, validation_split=0.2, callbacks=callbacks_list)
    # ===================================================

if __name__ == "__main__":   
    folder = './dataset/BlowingBubbles_416x240_50/'
    b = 16 # blk_size
    train_start, train_end = 0, 100
    coarse16_train(train_start,train_end, folder)