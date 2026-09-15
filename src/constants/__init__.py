from pathlib import Path

RANDOM_STATE = 42
# Mandatory constants for metadata formation
DATA_DIR = Path('data')
ZIPPED_DATA_FILE = Path(DATA_DIR, 'uw-madison-gi-tract-image-segmentation.zip')
EXTRACTED_DATA_DIR = Path(DATA_DIR, 'extracted_data')
PREPROCESSED_DATA_DIR = Path(DATA_DIR, 'preprocessed_data')
METADATA_FILE = Path(PREPROCESSED_DATA_DIR, 'final_dataframe.csv')
IMAGES_DIR = Path(EXTRACTED_DATA_DIR, 'train')


#Mandatory constants for image preparation 
_FINAL_DATA_DIR = Path(PREPROCESSED_DATA_DIR, 'final_data')
TRAIN_CSV = Path(_FINAL_DATA_DIR, 'train.csv')
DEV_CSV = Path(_FINAL_DATA_DIR, 'dev.csv')
TEST_CSV = Path(_FINAL_DATA_DIR, 'test.csv')
MANDAT_COLS = ['collapsed_id','case','slice','height','width','empty_slice','path','prev_path','next_path','large_bowel','small_bowel','stomach']

INP_IMG_DIR = Path(_FINAL_DATA_DIR, 'input_images')
OUT_MASK_DIR = Path(_FINAL_DATA_DIR, 'output_mask')


#Mandatory constants for image preparation 
MLFLOW_DATABASE_URL = 'sqlite:///mlflow.db' #TODO: SET THIS TO SOMETHING OR LEAVE AS IT IS. parameter for optuna study to store the data
OPTUNA_DATABASE_URL = 'sqlite:///optuna.db'
CHECKPOINT_DIR = Path('models', 'checkpoints') #TODO: SET THIS TO SOMETHING OR LEAVE AS IT IS. can be dir in database or local storage (pl.Callback parameter)

MODEL_NAME = 'MRI-Unet' # general model name for different versions e.g. MRI-Unet-1 
NUMBER_OPTUNA_TRIALS = 1  # optuna study parameter #! change this
BATCH_SIZE = 8 # dataloader parameter #! set to 32
NUM_WORKERS = 14 # dataloader parameter #!  SET TO NONE
DEVICES = 1 # trainer parameter  #!  SET TO NONE
ACCELERATOR = 'cpu' # trainerr parameter  #!  SET TO NONE
NUM_EPOCHS = 65  # lightning trainer parameter
LEARNING_RATE = 0.001 # opitmizer parameter
WEIGHT_DECAY = 0.01 # optimizer parameter
INPUT_CHANNELS = 3 # model input parameter
NUM_CLASSES = 3 # model output parmaeter
LABELS = {
    0: ['red', 'small_bowel'],
    1: ['green', 'large_bowel'],
    2: ['blue', 'stomach']
}