# Set rtx_optimizer to turn on RTX 16bits float optimization
rtx_optimizer = True

# =========================================================================
train_start, train_end = 0, 100
n_train_frames = train_end - train_start

test_start, test_end = 100, 200
n_test_frames = test_end - test_start

training_ratio = 6
testing_ratio =  6
# ============================== Directories ==============================
from pathlib import Path
# Get the absolute directoty (Unix Style) for SPIEcode-Daniel
root_dir = str(Path().resolve()).replace('\\', '/')


# dataset = "BasketballDrill_832x480_50"
dataset = "BlowingBubbles_416x240_50"

dataset_dir = root_dir + "/dataset/" + dataset
data_dir = dataset_dir + "/" 
save_dir = dataset_dir + "_residue8result/"

model_dir = root_dir + "/models/" + dataset + "/"

# ================================= Model =================================
def get_channel_strides(ratio):
    switcher={
        6 :{(2,1), 1},
        8 :{(2,4), 3},
        12:{(2,4), 2},
        24:{(4,4), 2},
        48:{(4,4), 1}
    }

    return switcher.get(ratio,"Invalid Ratio")

# ============================== Model Paths ==============================
# path example: C:\Users\danni\Documents\GitHub\SPIEcode-Daniel\models\BlowingBubbles_416x240_50\8\hdf5\
# path format: C:\Users\danni\Documents\GitHub\SPIEcode-Daniel\models\BlowingBubbles_416x240_50\ + 8 + \hdf5\ + model + .hdf5
def get_model_path(model, ratio):

    json_path = model_dir + str(ratio) + "/json/" + model + ".json"
    hdf5_path = model_dir + str(ratio) + "/hdf5/" + model + ".hdf5"

    return json_path, hdf5_path

# ========================== Training Parameters ==========================

def get_training_parameter(model):

    delta_switcher = {
        "coarse":           1.0,
        "prediction":       1.0,
        "prediction_gnet":        1.0,
        "prediction_gnet2":       1.0,
        "prediction_gnet3":       1.0,
        "prediction_gnet4":       1.0,
        "prediction_gnet5":       1.0,
        "prediction_gnet6":       1.0,
        "prediction_gnet8":       1.0,
        "residue":          1.0,
        "prediction_b1":    1.0,
        "prediction_b23":   1.0,
        "residue_b1":       1.0,
        "residue_b23":      1.0
    }
    delta = delta_switcher.get(model, "Invalid Model")

    patience_switcher = {
        "coarse":           500,
        "prediction":       100,
        "prediction_gnet":        100,
        "prediction_gnet2":       100,
        "prediction_gnet3":       100,
        "prediction_gnet4":       100,
        "prediction_gnet5":       100,
        "prediction_gnet6":       100,
        "prediction_gnet8":       100,
        "residue":          500,
        "prediction_b1":    10000,
        "prediction_b23":   10000,
        "residue_b1":       2000,
        "residue_b23":      500
    }
    n_patience = patience_switcher.get(model, "Invalid Model");

    batch_switcher = {
        "coarse":           256,
        "prediction":       128,
        "prediction_gnet":       128,
        "prediction_gnet2":      128,
        "prediction_gnet3":      128,
        "prediction_gnet4":      128,
        "prediction_gnet5":      128,
        "prediction_gnet6":      128,
        "prediction_gnet8":      64,
        "residue":          2048,
        "prediction_b1":    128,
        "prediction_b23":   128,
        "residue_b1":       512,
        "residue_b23":      512
    }
    batch_size = batch_switcher.get(model, "Invalid Model")

    epoch_switcher = {
        "coarse":           2000,
        "prediction":       2000,
        "prediction_gnet":        2000,
        "prediction_gnet2":       2000,
        "prediction_gnet3":       2000,
        "prediction_gnet4":       2000,
        "prediction_gnet5":       2000,
        "prediction_gnet6":       2000,
        "prediction_gnet8":       2000,
        "residue":          2000,
        "prediction_b1":    10000,
        "prediction_b23":   2000,
        "residue_b1":       2000,
        "residue_b23":      2000
    }
    epoch_size = epoch_switcher.get(model, "Invalid Model")
    # epoch_size = 1

    return  delta, n_patience, batch_size, epoch_size
