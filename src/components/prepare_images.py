from src.entity.config_entity import ImagePrepEntity
from src.logging import logger
from src.utils import *
import pandas as pd
import numpy as np
import ast


class ImagePreparation:
    def __init__(self, conf):
        self.conf = conf

    def __call__(self):
        metadata_df = pd.read_csv(self.conf.processed_metadata_file, converters={'large_bowel': ast.literal_eval, 'small_bowel': ast.literal_eval, 'stomach': ast.literal_eval})
