import os
import numpy as np
import gc
import time
import torch
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from src.logging import logger
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', None) is not None:
    from src.constants.kaggle import *
    

else:
    from src.constants.local import *
from src.components.optuna_model_selection.dataset import MRIDataset
from src.components.optuna_model_selection.model import MRIFlexAttentionUNet
from src.components.optuna_model_selection.lightning_modules import MRIModule
from tqdm.auto import tqdm
from lightning.pytorch.accelerators import TPUAccelerator

import optuna






class MetaDataEntity:
    def __init__(self):
        
        self.extr_data_dir = EXTRACTED_DATA_DIR
        self.train_dir = IMAGES_DIR
        self.train_csv_file = Path(EXTRACTED_DATA_DIR, 'train.csv')
        self.processed_metadata_file = METADATA_FILE
        self.processed_data_dir = PREPROCESSED_DATA_DIR

        logger.logging.info('Metadata entity successfully created')




class ImagePrepEntity:
    def __init__(self):

        self.processed_metadata_file = METADATA_FILE
        self.train_map_file = TRAIN_CSV
        self.dev_map_file = DEV_CSV
        self.test_map_file = TEST_CSV
        self.mandat_cols = MANDAT_COLS
        self.out_mask_dir = OUT_MASK_DIR
        self.random_state = RANDOM_STATE
        
        logger.logging.info('Image preparation entity successfully created')





class ModelCreationEntity:


    '''def __init__(self, accelerator: str|None = None, 
                 devices: int|None = None,  
                 num_workers: int|None = None, 
                 safety_batch =True):'''

    def __init__(self,
                 safety_batch =True):#!set to true


        self.lr = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY      
        self.inp_channels = INPUT_CHANNELS
        self.num_classes = NUM_CLASSES
        self.batch_size = BATCH_SIZE
        self.accelerator = ACCELERATOR
        self.devices = DEVICES
        self.num_workers = NUM_WORKERS
        self.train_csv = TRAIN_CSV
        self.dev_csv = DEV_CSV
        self.test_csv = TEST_CSV
        self.random_state = RANDOM_STATE
        self.model_name = MODEL_NAME
        self.checkpoint_dir = CHECKPOINT_DIR
        self.optuna_epochs = OPTUNA_EPOCHS
        self.final_epochs = FINAL_EPOCHS
        self.n_trials = NUMBER_OPTUNA_TRIALS
        self.mlflow_database_url = MLFLOW_DATABASE_URL
        self.optuna_database_url = OPTUNA_DATABASE_URL
        self.labels = LABELS
        self.pos_weight = POS_WEIGHT

        self.dev_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(256)
            ])
        
        
        self.train_transform = v2.Compose([
            v2.Resize(280),
            v2.RandomCrop(256),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.RandomRotation(15),
            v2.RandomApply([v2.ElasticTransform()], p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5)
            ])
        

        
        if not self.accelerator:
            self.accelerator = self.accelerator_test()
        


        if not self.devices: 
            self.devices = self.devices_test()
        

        if safety_batch:
            self.safe_batch_size_test()
        
        if self.num_workers is None:
            self.num_workers, _ = self.num_workers_test()

        if self.accelerator == 'tpu':
            self.drop_last_batch = True
        else:
            self.drop_last_batch = False



        if self.accelerator == 'gpu':
            self.strategy = GPU_DDP
        else:
            self.strategy = 'auto'


        

        logger.logging.info('Model creation entity successfully created')



    def accelerator_test(self):
        logger.logging.info(f'accelerator not passed as argument.')
        device = 'gpu' if torch.cuda.is_available() else None
        if not device:
            device = 'tpu' if TPUAccelerator.is_available() else 'cpu'
        logger.logging.info(f'Tested accelerator: {device}')
        print(f'Tested accelerator: {device}')
        return device



    def devices_test(self):
        logger.logging.info(f'Number of devices not passed as argument.')
        if self.accelerator == 'gpu':
            num = torch.cuda.device_count()

        #elif self.accelerator == 'tpu':
            #import torch_xla.core.xla_model as xm
            #num = len(xm.get_xla_supported_devices())

        else:
            num = torch.cpu.device_count()
        logger.logging.info(f'Tested number of devices: {num}')
        print(f'Tested number of devices: {num}')
        return num
    


    def num_workers_test(self, verbose = False, 
                         early_stopping = False, 
                         num_tested_batches = 52, warmup = 2 ):

        print(f"Benchmarking optimal number of workers...")
        logger.logging.info(f'number of workers not passed as argument.')

        from src.components.optuna_model_selection.dataset import MRIDataset

        if verbose:
            stats = {}
        
        dataset = MRIDataset(self.train_csv, transform = self.train_transform)

        if self.accelerator == 'gpu':
            device = torch.device('cuda')

        #elif self.accelerator == 'tpu':
            #import torch_xla.core.xla_model as xm
            #device = xm.xla_device()

        else:
            device = torch.device('cpu')

        was_better = True

        #device = xm.xla_device()
        best_time = float('inf')
        best_worker = 0

        for workers in range(0, os.cpu_count()+1, 2):
            wait_times, work_times = [], []

            dataloader = DataLoader(dataset, self.batch_size, True, num_workers=workers)

            end_of_prev = time.time()

            try:
                data_iter = iter(dataloader)

                for batch_idx in range(num_tested_batches):

                    data = next(data_iter)
                    start_of_current = time.time()
                    wait_times.append(start_of_current - end_of_prev)

                    img, mask = data
                    img = img.to(device)
                    mask = mask.to(device)

                    if self.accelerator == 'gpu':
                        torch.cuda.synchronize()

                    end_of_current = time.time()
                    curr_time = end_of_current - start_of_current
                    work_times.append(curr_time)
                                    
                    end_of_prev = end_of_current

                if len(work_times) > warmup:
                    avg_work = np.mean(work_times[warmup:])
                    avg_wait = np.mean(wait_times[warmup:])
                    act_time = avg_work + avg_wait

                    print(f'workers: {workers}, time: {act_time:.3f}')

                    if act_time < best_time:
                        best_time = act_time
                        best_worker = workers

                        was_better = True

                    else: 
                        if early_stopping and not was_better:
                            break
                        was_better = False

                    

                    if verbose:
                        stats[workers] = [avg_wait, avg_work, act_time]

                    
                    


            except StopIteration:
                print('super')

            except RuntimeError as e:
                print(2)

            finally:
                del dataloader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f'The optimal number of effective workers: {best_worker}')
        logger.logging.info(f'Tested nuber of workers: {best_worker}')

        if verbose:
            return best_worker // self.devices, best_time, dict(sorted(stats.items(), key=lambda item: item[1][-1]))
                
        return best_worker // self.devices, best_time



    def safe_batch_size_test(self):

        print(f"Benchmarking optimal batch size...")
        logger.logging.info(f'Batch size of {self.batch_size} . Checking whether it is valid...')
        if self.accelerator == 'gpu':
            device = torch.device('cuda')

        #elif self.accelerator == 'tpu':
          
            #import torch_xla.core.xla_model as xm
            #device = xm.xla_device()

        else:
            device = torch.device('cpu')
        
        model = MRIFlexAttentionUNet(self.inp_channels, 64, self.num_classes, 4, 3, 3, 5 ).to(device) #established max model hyperparameters
        
        model.train()
        
        current_batch = self.batch_size
        safe_batch = 0
        
        while current_batch > safe_batch:
            try:
                dummy_img, outputs, loss = None, None, None

                dummy_img = torch.randn(current_batch, self.inp_channels, 256, 256, device=device)
                
                outputs = model(dummy_img)
                loss = outputs.sum()
                loss.backward()
                
                model.zero_grad()

                logger.logging.info(f"Batch size {current_batch} is valid")

                print(f"Batch size {current_batch} is valid")
                safe_batch = current_batch
    
                
            except RuntimeError as e:
                # 3. Catch the specific PyTorch CUDA OOM error
                if "out of memory" in str(e).lower():
                    print(f"Batch size {current_batch} is not valid. Testing batch size of {current_batch//2}")
                    current_batch = current_batch // 2
                else:
                    raise e # Re-raise if the error is unrelated to memory
            finally:
                del dummy_img, outputs, loss

                if self.accelerator == 'gpu':
                    torch.cuda.empty_cache()    
        
        del model
        torch.cuda.empty_cache()

        if safe_batch == 0:
            raise RuntimeError("Even batch size 1 resulted in OOM. Model is too large for this GPU.")
        
        self.batch_size = safe_batch
    








class SelectedModelsCreationEntity(ModelCreationEntity):


    def __init__(self, index):
        super().__init__(safety_batch=False)
        
        self.checkpoint_dir = CHECKPOINT_V2_DIR
        self.epochs = V2_EPOCHS
        
        self.index = index

        if index > 100:
            self.first_conv_out_channels = 64
            self.depth = 5
            self.n_encoder_conv_layers = 2
            self.n_decoder_conv_layers = 2
            self.kernel_sizes = [5,3,5,3,3,3,3,5,3,5]
            self.empty_mri_ratio = 0.1
            self.lr = 0.0005
            self.w = 0.005
            self.ckpt_inp_model_pathes=None


        else:        
            study = optuna.load_study(study_name='mri_unet_optuna_search_v2-1: more models less epochs', storage = OPTUNA_DATABASE_URL)
            df = study.trials_dataframe()

            df = df[df['value'].notna()]
            df = df.sort_values(axis = 0, by = 'value', ignore_index=True )
            df = df.drop(['datetime_start', 'datetime_complete', 'duration'],axis = 1)


            self.first_conv_out_channels = int(df.iloc[index]['params_first_conv_out_channels'])
            self.depth = int(df.iloc[index]['params_depth'])
            self.n_encoder_conv_layers=int(df.iloc[index]['params_n_encoder_conv_layers'])
            self.n_decoder_conv_layers = int(df.iloc[index]['params_n_decoder_conv_layers'])
            self.kernel_sizes = [int(df.iloc[index][f'params_kernel_sizes_{x}']) for x in range(self.depth*2)]
            self.empty_mri_ratio = df.iloc[index]['params_empty_mri_ratio']
            self.lr = df.iloc[index]['params_learning_rate_start']
            self.w=df.iloc[index]['params_weight decay']
            self.ckpt_inp_model_pathes=None #Path(f'models/v2_inp/model_{index}.ckpt')
            logger.logging.info('Model creation entity for second version of search created')


from src.components.prepare_images import add_data

class FinalModelCreationEntity(ModelCreationEntity):


    def __init__(self):
  
        super().__init__(safety_batch=False)
        add_data(TRAIN_CSV, DEV_CSV, FINAL_CSV)
        self.train_csv = FINAL_CSV

        self.dev_csv = TEST_CSV
        
        self.checkpoint_dir = FINAL_MODEL_CHECKPOINT
        self.epochs = FINAL_EPOCHS
        self.inp_model_path = INPUT_CHECKPOINT_FOR_FINAL_MODEL
        model = MRIModule.load_from_checkpoint(INPUT_CHECKPOINT_FOR_FINAL_MODEL)
        hparams = model.hparams
        del model
        torch.cuda.empty_cache()

        self.first_conv_out_channels = hparams['first_conv_out_channels']
        self.depth = hparams['depth']
        self.n_encoder_conv_layers = hparams['n_encoder_conv_layers']
        self.n_decoder_conv_layers = hparams['n_decoder_conv_layers']
        self.kernel_sizes = hparams['kernel_sizes']
        self.empty_mri_ratio = hparams['empty_mri_ratio']
        self.lr = hparams['learning_rate']
        self.w = hparams['weight_decay']
        

        logger.logging.info('Final model creation entity created')