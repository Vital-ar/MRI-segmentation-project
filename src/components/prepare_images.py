from src.entity.config_entity import ImagePrepEntity
from src.logging import logger
from src.utils import *
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import os






class ImagePreparation:
    def __init__(self, conf: ImagePrepEntity):
        self.conf = conf
        self.rng = np.random.default_rng(seed = self.conf.random_state)
        if self.conf.processed_metadata_file.exists():
            try:
                metadata_df = pd.read_csv(self.conf.processed_metadata_file)
                logger.logging.info(f'successfully retrieved meta data file from: {self.conf.processed_metadata_file}')
                valid = np.prod([c in metadata_df.columns for c in self.conf.mandat_cols ])
                if valid:
                    logger.logging.info(f'successfully confirmed meta data file from: {self.conf.processed_metadata_file}')
                    self.metadata_df = metadata_df
                else:
                    logger.logging.error(f'rejected meta data file from: {self.conf.processed_metadata_file}')
                    raise ValueError()

            except Exception as e:
                logger.logging.error(f'error occured while trying to read csv file at: {self.conf.processed_metadata_file}:\n{e}')
                raise e
        else:
            logger.logging.error(f'there is no such file: {self.conf.processed_metadata_file}')
            raise FileNotFoundError
        




    def _make_inp_out_map(self, fraction: int, cases:np.ndarray, file_path: Path) ->np.ndarray:
        try:
            if not file_path.parent.exists():

                os.makedirs(file_path.parent)
                logger.logging.info(f'successfully created directory: {file_path.parent}')

            if not file_path.exists():

                logger.logging.info(f'file: {file_path} - does not exist. creating new one...')

                if len(cases) > fraction:
                    chousen_cases = self.rng.choice(cases, fraction)
                    chousen_cases.sort()
                    
                    
                    avail_cases = [c for c in cases if c not in chousen_cases]
                else:
                    chousen_cases = cases
                    avail_cases = None

                final_df = self.metadata_df[self.metadata_df.case.isin(chousen_cases)].reset_index(drop = True)
                final_df = final_df.drop('case', axis = 1)

                final_df.to_csv(file_path, index = False)

                logger.logging.info(f'successfully created csv file at: {file_path}')

            else:
                logger.logging.info(f'file: {file_path} - already exists')
                avail_cases = cases

            return avail_cases
        except Exception as e:
            logger.logging.error(f'error at _make_inp_out_map:\n{e}')
            print(e) 
            raise e

    

    def __call__(self, dev_ratio: float = 0.05, test_ratio: float = 0.05):

        try:
            #masks' folder creation
            self.metadata_df['mask_path'] = self.metadata_df.apply(lambda n: Path(self.conf.out_mask_dir, f'{n['collapsed_id']}_{n['case']}_{n['slice']}.npy' ), axis = 1)
            os.makedirs(self.conf.out_mask_dir, exist_ok= True)

            num_files = len(os.listdir(self.conf.out_mask_dir))
            if num_files > 0:
                logger.logging.info(f'directory: {self.conf.out_mask_dir} - already hase {num_files} inner files, skiping creation of masks')

            else:
                logger.logging.info(f'directory: {self.conf.out_mask_dir} - hase no inner files, entering creation of masks...')
                num_masks = 0
                for row in self.metadata_df.itertuples(index = True):
                    #! LABEL ORDER
                    mask = make_img_mask(row.large_bowel, row.small_bowel, row.stomach, row.height, row.width)
                    num_masks += 1
                    np.save(row.mask_path, mask)

                logger.logging.info(f'directory: {self.conf.out_mask_dir} - successfully filled with {num_masks} masks')

            
            if self.conf.train_map_file.exists() & self.conf.dev_map_file.exists() & self.conf.test_map_file.exists():
                logger.logging.info(f'train, dev and test csv files already exists at:\n{self.conf.train_map_file}\n{self.conf.dev_map_file}\n{self.conf.test_map_file}')


            else:
                #train, dev and test mapping files creation
                logger.logging.info(f'files: \n{self.conf.train_map_file}\n{self.conf.dev_map_file}\n{self.conf.test_map_file}\n not found. Entering creation of train, dev and test csv files')

                self.metadata_df = self.metadata_df.drop(['collapsed_id', 'slice', 'height', 
                                                          'width', 'large_bowel', 'small_bowel', 'stomach'], axis = 1)

            
                cases = self.metadata_df.case.unique()
                case_len = len(cases)
               
                dev_fract = int(np.round(dev_ratio * case_len))
                test_fract = int(np.round(test_ratio * case_len))
                train_fract = case_len - dev_fract - test_fract
                
                avail_cases = self._make_inp_out_map(dev_fract, cases, self.conf.dev_map_file)

                train_cases = self._make_inp_out_map(test_fract, avail_cases, self.conf.test_map_file)

                _ = self._make_inp_out_map(train_fract, train_cases, self.conf.train_map_file)

                logger.logging.info(f'train, dev and test csv files successfully created')


        except Exception as e:
            logger.logging.error(f'Error: {e}')
            raise e
                




        

        


        