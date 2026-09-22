import optuna
from src.components.optuna_model_selection.model_trainer import ModelTrainer
from src.entity.config_entity import SelectedModelsCreationEntity
import gc
import torch
from src.logging import logger



class V2TtrainingComponent:
    def __init__(self, config: SelectedModelsCreationEntity):

       
        self.config = config
        self.counter = 27
        
        model_trainer = ModelTrainer(
            trial = None,
            learning_rate = self.config.lr,
            weight_decay = self.config.w,                     
            inp_channels = self.config.inp_channels,
            first_conv_out_channels = self.config.first_conv_out_channels,
            num_classes = self.config.num_classes,
            depth = self.config.depth,
            n_encoder_conv_layers = self.config.n_encoder_conv_layers,
            n_decoder_conv_layers = self.config.n_decoder_conv_layers,
            kernel_sizes = self.config.kernel_sizes,
            
            train_csv = self.config.train_csv,
            dev_csv= self.config.dev_csv,
            test_csv = self.config.test_csv,
            batch_size = self.config.batch_size, # defined by test in config entity
            num_workers = self.config.num_workers, # defined by test in config entity
            train_transform=self.config.train_transform,
            dev_transform=self.config.dev_transform,
            train_empty_mri_ratio = self.config.empty_mri_ratio, 
            random_state=self.config.random_state,
            model_name = f'{self.config.model_name}-{self.config.index}',
            database_url=self.config.mlflow_database_url,
            checkpoint_dir= self.config.checkpoint_dir,
            ckpt_path= self.config.ckpt_inp_model_pathes
        )

        
        
    
    
    def __call__(self):
        logger.logging.info('Start of optuna opitmization process...')
        best_val_loss = model_trainer(
            num_epochs = self.config.optuna_epochs,
            accelerator= self.config.accelerator, # defined by test in config entity
            devices = self.config.devices # defined by test in config entity
        )
        logger.logging.info(f'model {self.config.model_name}-{self.config.index} successfully trained')
                
                
        del model_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        
        logger.logging.info('Optuna optimization successfully finished')