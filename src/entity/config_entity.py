from datetime import datetime
import os
from src.logging import logger
from dataclasses import dataclass
from pathlib import Path
from src.constants.metadata_stage import *


class MetaDataEntity:
    def __init__(self):
        logger.logging.info('Metadata entity successfully created')
        self.extr_data_dir = EXTRACTED_DATA_DIR
        self.train_dir = Path(EXTRACTED_DATA_DIR, 'train')
        self.train_csv_file = Path(EXTRACTED_DATA_DIR, 'train.csv')
        self.processed_metadata_file = METADATA_NAME
        self.processed_data_dir = PREPROCESSED_DATA_DIR
        



class ImagePrepEntity:
    def __init__(self):
        self.processed_metadata_file 


        self.train_folder
        self.dev_folder
        self.test_folder
        self.inp_arr_file
        self.out_mask_file
        self.in_out_map



