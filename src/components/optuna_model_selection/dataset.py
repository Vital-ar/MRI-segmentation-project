import torch
from torchvision import tv_tensors, transforms
from torch.utils.data import Dataset
from pathlib import Path

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













class MRISaverDataset(Dataset):



    def __init__(self, subset_df, transform = None, random_state =42):
        logger.logging.info('entering the creation of dataset')
        df = subset_df[~subset_df.empty_slice].drop('empty_slice', axis = 1)
        self.empty_df = subset_df[subset_df.empty_slice].drop('empty_slice', axis = 1)

        
        self.transform = transform
        self.random_state = random_state

        self.partial_df = subset_df

    
    def __len__(self):

        return len(self.partial_df)

         

    def __getitem__(self, index): 
        
        entry = self.partial_df.iloc[index]

        save_p = Path(entry.out_transformed_mask_path)
        if not save_p.parent.is_dir():
            save_p.parent.mkdir(parents=True, exist_ok = True)
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
            if not save_p.is_file():
                np.savez_compressed(save_p, mask.to(torch.bool).cpu().numpy())
        

        return img, index

    

   

class CorrectionDataset(Dataset):
    def __init__(self, subset_df_path, empty_img_ratio = 0.1, transform = None, random_state = 42, max_epoch = 200, max_seed = 2000):
        logger.logging.info('entering the creation of correction dataset')
        subset_df = pd.read_csv(subset_df_path).drop(['collapsed_id', 'case', 'slice', 'prev_path', 'next_path'], axis = 1)

        self.not_empty_df = subset_df[~subset_df.empty_slice].drop('empty_slice', axis = 1)
        self.empty_df = subset_df[subset_df.empty_slice].drop('empty_slice', axis = 1)

        self.empty_ratio = empty_img_ratio
        self.transform = transform
        self.random_state = random_state
        summed_norm = np.array([74,61,45])
        self.summed_norm = np.expand_dims(summed_norm, (1,2))

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

        
        img = cv2.imread(entry.path, cv2.IMREAD_UNCHANGED)
        inp_mask = np.load(entry.model_mask_path, allow_pickle=True)['arr_0']
        out_mask = np.load(entry.out_transformed_mask_path, allow_pickle=True)['arr_0']
        summed_mask = np.load(entry.collapsed_mask_path, allow_pickle=True)['arr_0']
    
        

        min_img = img.min()
        max_img = img.max()
        img = (img - min_img)/(max_img - min_img + 1e-8)
        summed_mask = summed_mask/self.summed_norm

        img = torch.from_numpy(img).to(torch.float32)
        inp_mask = torch.from_numpy(inp_mask).to(torch.float32)
        out_mask = torch.from_numpy(out_mask).to(torch.float32)
        summed_mask = torch.from_numpy(summed_mask).to(torch.float32)

        if img.ndim == 2:
            img = img.unsqueeze(0)

        if self.transform:
            img = self.transform(img)

     

        return (img, inp_mask, summed_mask), out_mask

    

    def on_epoch_end(self):

        if (self.empty_ratio < self.orig_rat) & (self.empty_ratio > 0):
            if self.idx == len(self.seed_arr):
                        self.idx = 0
            self.partial_df = self._get_part_df(self.seed_arr[self.idx])
            self.idx += 1




