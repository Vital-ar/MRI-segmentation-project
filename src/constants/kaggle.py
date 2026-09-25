from pathlib import Path
import os

RANDOM_STATE = 42
# Mandatory constants for metadata formation
DATA_DIR = Path('data')
ZIPPED_DATA_FILE = Path(DATA_DIR, 'uw-madison-gi-tract-image-segmentation.zip')
EXTRACTED_DATA_DIR = Path(DATA_DIR, 'extracted_data')
PREPROCESSED_DATA_DIR = Path('/kaggle/input/mri-segmentation-dataset', 'preprocessed_data')
METADATA_FILE = Path(PREPROCESSED_DATA_DIR, 'final_dataframe.csv')
IMAGES_DIR = Path(EXTRACTED_DATA_DIR, 'train')


#Mandatory constants for image preparation 
_FINAL_DATA_DIR = Path(PREPROCESSED_DATA_DIR, 'final_data')
TRAIN_CSV = Path('data_csv', 'train_kaggle.csv')
DEV_CSV = Path('data_csv', 'dev_kaggle.csv')
TEST_CSV = Path('data_csv', 'test_kaggle.csv')
MANDAT_COLS = ['collapsed_id','case','slice','height','width','empty_slice','path','prev_path','next_path','large_bowel','small_bowel','stomach']

INP_IMG_DIR = Path(_FINAL_DATA_DIR, 'input_images')
OUT_MASK_DIR = Path(_FINAL_DATA_DIR, 'output_mask')


#Mandatory constants for image preparation 
MLFLOW_DATABASE_URL = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db' )
OPTUNA_DATABASE_URL = 'sqlite:///optuna.db'
CHECKPOINT_DIR = Path('models', 'checkpoints') 

MODEL_NAME = 'MRI-Unet_v3-1' # general model name for different versions e.g. MRI-Unet-1 
NUMBER_OPTUNA_TRIALS = 40  # optuna study parameter #! change this
BATCH_SIZE = 4 # dataloader parameter #! set to 32
NUM_WORKERS = 0 # dataloader parameter #!  SET TO NONE
DEVICES = 8 # trainer parameter  #!  SET TO NONE
ACCELERATOR = 'tpu' # trainerr parameter  #!  SET TO NONE
OPTUNA_EPOCHS = 4 # lightning trainer parameter
FINAL_EPOCHS = 40
LEARNING_RATE = 0.001 # opitmizer parameter
WEIGHT_DECAY = 0.001 # optimizer parameter
INPUT_CHANNELS = 3 # model input parameter
NUM_CLASSES = 3 # model output parmaeter
LABELS = {
    0: ['red', 'small_bowel'],
    1: ['green', 'large_bowel'],
    2: ['blue', 'stomach']
}




CHECKPOINT_V2_DIR = Path('models', 'v2_out') 
V2_EPOCHS = 30





GPU_DDP = 'ddp_spawn'# optuna crash on auto 



#* calculated in /exploritary/optuna_exp.ipynb end section 
#* (parameter for bce loss calculated for full train dataset, and for partial train dataset with empty img ratio: 0.1, 0.2, 0.3)

POS_WEIGHT_FULL = [139.66749294, 152.21774755, 272.26512968]
POS_WEIGHT_0_1 = [ 66.76699651,  72.81312019, 130.64631503]
POS_WEIGHT_0_2 = [ 75.23487254,  82.03649415, 147.09627935]
POS_WEIGHT_0_3 = [ 86.12556861,  93.89885046, 168.25289069]
POS_WEIGHT = POS_WEIGHT_0_1
EFFECTIVE_BATCH_SIZE = 32

V3_EPOCHS = 90