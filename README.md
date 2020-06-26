# Motion-Aware Deep Video Coding Network

Refactored and updated by **Dannier Li (Chlerry)** for further research between *03/20/2020* and *06/26/2020*

Initially created by **Ying Liu** on *02/12/2020*

**Paper Reference:**  
R. Khan, Y. Liu, "Motion-aware deep video coding network," Proceedings Volume 11395, *Big Data II: Learning, Analytics, and Applications*; 113950B (2020) [https://doi.org/10.1117/12.2560814](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11395.toc)  
Event: SPIE Defense + Commercial Sensing, 2020, Online Only


***Santa Clara University, Santa Clara, CA***  

## File Structure

```
└── SPIECode-Daniel
    ├── dataset
    │   └── BlowingBubbles_416x240_50
    ├── models
    |   └── BlowingBubbles_416x240_50
    |       └── 6 (compression ratio)
    |           ├── hdf5
    |           └── json
    ├── utility      
    |   ├── __init__.py 
    |   ├── helper.py
    |   └── parameter.py    
    ├── coarse     
    |   ├── __init__.py 
    |   ├── train.py
    |   └── test.py
    ├── prediction
    |   ├── __init__.py 
    |   ├── train.py
    |   ├── inference.py
    |   ├── b1_train.py
    |   ├── b1_inference.py
    |   ├── b23_train.py
    |   ├── b23_inference.py
    |   └── train_gnet5_new.py
    ├── residue
    |   ├── __init__.py 
    |   ├── train.py
    |   ├── inference.py
    |   ├── b1_train.py
    |   ├── b23_train.py
    |   └── b_inference.py
    ├── __main__.py
    ├── spie.ipynb
    ├── spie_new.ipynb
    ├── plot.ipynb
    ├── results
    └── README.md
```

## Environment Setup  
The project is developed on `Windows 10.19041` and `Ubuntu 18.04` with `conda 4.8.3`. Code could be running on `Anaconda PowerShell`, `Ubuntu` and `Spyder 4`.  
Package dependencies should be installed in `Anaconda` with following command lines in `Ubuntu Terminal` or `PowerShell` (Open `PowerShell` as **Administrator** if you are using `Windows 10`).  
> conda install -c anaconda tensorflow-gpu  
> conda install -c conda-forge keras  
> conda install -c conda-forge scikit-learn  
> conda install -c conda-forge scikit-image  
> conda install -c conda-forge numpy  
> conda install -c conda-forge opencv

## Training Configuration: Parameters
All training and testing parameters can be found and changed in `utility/parameters.py`. Some important parameters are listed below: 
- Turn off RTX Float16 Optimizer: set `rtx_optimizer` to `False`
- Update the range of taining data and testing data: `train_start`, `train_end`, `test_start`, `test_end`   
- Update compression ratio: `training_ratio`, `testing_ratio`  
- Update `min_delta`, `patience`, `batch` size, and `epoch` size in: 
    > get_training_parameter(model)

## Running with Spyder 4
- Open `SPIEcode-Daniel` as a `Spyder` project
- Set the working directory to `SPIEcode-Daniel`
- Open and run `./__main__.py` with `python` in `Powershell`
- Uncomment the following code to train **original** model
    - Train compression net  
        > coarse.train.main()  
    - Train prediction net  
        > prediction.train.main()  
    - Train residue net  
        > residue.train.main()  
- Uncomment the following code to test **original** model
    - Test compression net  
        > coarse.test.main()  
    - Test prediction net  
        > prediction.inference.main()
    - Test residue net  
        > residue.inference.main()  
- Uncomment the following code to train **new** model
    - Train b1's prediction net (Since currently we are using b model to predict b1 frames, the training code remains the same as b prediction) 
        > ~~prediction.b1_train.main()~~  
        >   prediction.train.main()  
    - Train b1's residue net  
        > residue.b1_train.main()
    - Train b2 and b3's prediction net  
        > prediction.b23_train.main()  
    - Train b2 and b3's residue net   
        > residue.b23_train.main()
- Uncomment the following code to test **new** model  
    - Test b1's prediction net  
        > prediction.b1_inference.main()  
    - Test b2 and b3's prediction net  
        > prediction.b23_inference.main()  
    - Test b1, b2 and b3's residue net  
        > residue.b_inference.main()
- Uncomment the following code to train  **GNet5** model
    - Train b and b1's prediction net  
        > prediction.train_gnet5_new.main()  
- Test b's **GNet5** model  
  I would strong recommend using jupyter notebook or command line to run the code instead of the following. Anyway, if you are still using Spider:
    First, please update `prediction.inference.main()`: 
    > `--` regrouped_prediction = predict(decoded, b, bm, testing_ratio)  
    > `++` regrouped_prediction = predict(decoded, b, bm, testing_ratio, 'prediction_gnet5')  
    >  
  Then uncomment the following line  
    > prediction.inference.main()  

- Test b1's **GNet5** prediction 
  First, please first update `prediction.b1_inference.main()`:  
    > `--` predicted_b1_frame = pred_inference_b1(decoded, b, bm, training_ratio)  
    > `++` predicted_b1_frame = pred_inference_b1(decoded, b, bm, training_ratio, 'prediction_gnet5')  
    >  
  Then uncomment the following line  
    > prediction.b1_inference.main()


## Running with PowerShell or Jupyter Notebook 
1. The following source code are associated with the original method: 
    - `utility/*`
    - `spie.ipynb`
    - `coarse_train.py`; `coarse_test.py`
    - `prediction_train.py`; `prediction_inference.py`
    - `residue_train.py`; `residue_inference.py` 
2. The following models have been trained for the original method: 
    - `models/BlowingBubbles_416x240_50*`
3. Parameters can be changed in `utility/parameter.py` with following attributes:
    - RTX Optimizer
    - choice of datasets
    - directories
    - batch size
    - epoch size
4. Please note that if you are using GTX graphics instead of RTX graphics, change the `rtx_optimizer` in `parameters` to `false` to disable `float16` optimization
5. The simplest way to run the code is running `spie.ipynb` with `Jupyter Notebook`. However, the following command is avaible to train and run the model directly with `python`:
    - Testing on the test dataset for all models 
        > python -m residue.inference
    - Train models 
        > python -m coarse.train  
        > python -m prediction.train  
        > python -m residue.train


## Ignore Optimizer Warning
 > UserWarning: TensorFlow optimizers do not make it possible to access optimizer attributes
