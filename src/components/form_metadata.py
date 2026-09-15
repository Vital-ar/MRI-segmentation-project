from src.entity.config_entity import MetaDataEntity
import os
import pandas as pd
from src.utils import *
import re 
from src.logging import logger



class FormMetaDataComponent:
    def __init__(self, conf: MetaDataEntity):
        self.conf = conf

    def __call__(self, intermediate_full_df = None):
        if not os.path.exists(self.conf.processed_metadata_file):
            logger.logging.info(f'{self.conf.processed_metadata_file} does not exists. Entering metadata formation stage')
            if os.path.exists(self.conf.train_csv_file):
                try: 
                    inp_df = pd.read_csv(self.conf.train_csv_file)
                    path_id_dict = get_path_id_dict(self.conf.train_dir)   
                    path_id_df = pd.DataFrame(path_id_dict)
                    full_df = pd.merge(inp_df, path_id_df, how = 'inner') 
                    full_df['width'] = full_df['path'].apply(lambda p: int(p.parts[-1].split('_')[2]))
                    full_df['height'] = full_df['path'].apply(lambda p: int(p.parts[-1].split('_')[3]))
                    #full_df['segmentation'] = full_df['segmentation'].fillna(value = '0')
                    #full_df['segmentation'] = full_df['segmentation'].apply(lambda n: [int(k) for k in n.split()] if n != '0' else int(n))
                    print(type(full_df.iloc[38473]['segmentation']))
                    full_df = full_df.pivot(index = ['id', 'path', 'height', 'width'],
                               columns='class', 
                               values = 'segmentation').reset_index()
                    full_df.columns.name = None
                    
                    full_df['case'] = full_df['id'].apply(lambda n : int(re.findall(r'\d+', n)[0]))
                    full_df['day'] = full_df['id'].apply(lambda n : int(re.findall(r'\d+', n)[1]))
                    full_df['slice'] = full_df['id'].apply(lambda n: int(re.findall(r'\d+', n)[2]))

                    if intermediate_full_df:
                        full_df = full_df[['case', 'day', 'slice', 'id', 'path', 'height', 'width', 'large_bowel', 'small_bowel', 'stomach']]
                        
                        
                        os.makedirs(self.conf.processed_data_dir, exist_ok=True)
                        full_df.to_csv(Path(self.conf.processed_data_dir, intermediate_full_df ), index=False)
                        logger.info(f'{intermediate_full_df} successfully created')

                    collapsed_df = full_df
                    collapsed_df['collapsed_id'] = collapsed_df.groupby(['case', 'day']).ngroup()
                    collapsed_df = collapsed_df.drop(['day', 'id'], axis = 1)
                    collapsed_df = collapsed_df.sort_values(['collapsed_id', 'slice']).reset_index(drop=True)

                    collapsed_df['prev_path'] = collapsed_df.groupby('collapsed_id')['path'].shift(1)
                    collapsed_df['next_path'] = collapsed_df.groupby('collapsed_id')['path'].shift(-1)
                    collapsed_df['prev_path'] = collapsed_df['prev_path'].fillna(collapsed_df['path'])
                    collapsed_df['next_path'] = collapsed_df['next_path'].fillna(collapsed_df['path'])

                    collapsed_df['empty_slice'] = collapsed_df['large_bowel'].isna() & collapsed_df['small_bowel'].isna() & collapsed_df['stomach'].isna()

                    collapsed_df = collapsed_df[['collapsed_id', 'case', 'slice', 'height', 'width', 
                                                 'empty_slice', 'path', 'prev_path', 'next_path', 
                                                 'large_bowel','small_bowel','stomach']]

                    os.makedirs(self.conf.processed_data_dir, exist_ok=True)
                    collapsed_df.to_csv(self.conf.processed_metadata_file, index=False)
                    logger.logging.info(f'{self.conf.processed_metadata_file} successfully created')

                except Exception as e:
                    logger.logging.error(f'error occured while trying to create  meta data file: {e}')
                    raise e
            else:
                logger.logging.error(f'({self.conf.train_csv_file}) - the neccessary file for creating metadata file does not exist')
                raise FileNotFoundError
        else:
            logger.logging.info(f'{self.conf.processed_metadata_file} allready exists. Skiping metadata formation stage')
            


