import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset

from src.logging import logger
import pandas as pd
import numpy as np
import cv2







class MRIDataset(Dataset):



    def __init__(self, subset_df_path, empty_img_ratio = 0.2, transform = None, random_state = 42, max_epoch = 200, max_seed = 2000):
        logger.logging.info('entering the creation of dataset')
        subset_df = pd.read_csv(subset_df_path)

        self.not_empty_df = subset_df[~subset_df.empty_slice].drop('empty_slice', axis = 1)
        self.empty_df = subset_df[subset_df.empty_slice].drop('empty_slice', axis = 1)

        self.empty_ratio = empty_img_ratio
        self.transform = transform
        self.random_state = random_state


        rng = np.random.default_rng(seed = random_state)
        
        self.seed_arr = rng.integers(1, max_seed, max_epoch)
        self.idx = 0

        self.orig_rat = len(self.empty_df) / len(subset_df)

        if (self.empty_ratio < self.orig_rat) & (self.empty_ratio > 0):
            self.empty_fract = int(len(self.not_empty_df) / (1 - self.empty_ratio) * self.empty_ratio)
            self.partial_df = self._get_part_df(self.seed_arr[self.idx])
            self.idx = 1
            

        elif self.empty_ratio == 0:
            self.partial_df = self.not_empty_df
            self.empty_fract = 0

        else:
            self.partial_df = subset_df
            self.empty_fract = len(self.empty_df)

        logger.logging.info('dataset successfully created. DO NOT FORGT TO USE ---> on_epoch_end <--- FUNCTION')
          
        

    def _get_part_df(self, random_random_state):
        df = pd.concat(
            [self.not_empty_df, 
             self.empty_df.sample(frac = 1, random_state= random_random_state).iloc[:self.empty_fract]], 
             ignore_index=True)
        return df


    
    def __len__(self):

        return len(self.partial_df)

         

    def __getitem__(self, index): 

        entry = self.partial_df.iloc[index]
    
        img_0 = cv2.imread(entry.prev_path, cv2.IMREAD_UNCHANGED)
        img_1 = cv2.imread(entry.path, cv2.IMREAD_UNCHANGED)
        img_2 = cv2.imread(entry.next_path, cv2.IMREAD_UNCHANGED)

        img = np.stack([img_0, img_1, img_2])
        
        mask = np.load(entry.mask_path)

        img = tv_tensors.Image(torch.from_numpy(img)).to(torch.float32)
        mask = tv_tensors.Mask(torch.from_numpy(mask)).to(torch.float32)

        min_num = img.min()
        max_num = img.max()
        img = (img - min_num)/(max_num - min_num + 1e-8)
        
        if self.transform:
            img, mask = self.transform(img, mask)

        

        return img, mask

    

    def on_epoch_end(self):

        if (self.empty_ratio < self.orig_rat) & (self.empty_ratio > 0):
            if self.idx == len(self.seed_arr):
                        self.idx = 0
            self.partial_df = self._get_part_df(self.seed_arr[self.idx])
            self.idx += 1