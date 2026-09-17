import optuna
from src.components.optuna_model_selection.model_trainer import ModelTrainer
from src.entity.config_entity import ModelCreationEntity
import gc
import torch
from src.logging import logger



class OptunaModelSelectionComponent:
    def __init__(self, config: ModelCreationEntity):

        self.study = optuna.create_study(study_name= 'mri_unet_optuna_search_v2', 
                                         storage = config.optuna_database_url, 
                                         direction='minimize', load_if_exists=True)
        self.config = config
        self.counter = 0
        

    def _objective(self, trial: optuna.Trial):

        first_conv_out_channels = trial.suggest_categorical('first_conv_out_channels', [32,64])
        depth = trial.suggest_int('depth', 3, 4)
        n_encoder_conv_layers = trial.suggest_int('n_encoder_conv_layers',1, 3)
        n_decoder_conv_layers = trial.suggest_int('n_decoder_conv_layers',1, 3)
        kernel_sizes = [trial.suggest_categorical(f'kernel_sizes_{i}', [3, 5]) for i in range(depth*2)]
        empty_mri_ratio = trial.suggest_float('empty_mri_ratio', 0.1, 0.3, step = 0.05)
        
        model_trainer = ModelTrainer(
            trial = trial,
            learning_rate = self.config.lr,
            weight_decay = self.config.weight_decay,                     
            inp_channels = self.config.inp_channels,
            first_conv_out_channels = first_conv_out_channels,
            num_classes = self.config.num_classes,
            depth = depth,
            n_encoder_conv_layers = n_encoder_conv_layers,
            n_decoder_conv_layers = n_decoder_conv_layers,
            kernel_sizes = kernel_sizes,
            
            train_csv = self.config.train_csv,
            dev_csv= self.config.dev_csv,
            test_csv = self.config.test_csv,
            batch_size = self.config.batch_size, # defined by test in config entity
            num_workers = self.config.num_workers, # defined by test in config entity
            train_transform=self.config.train_transform,
            dev_transform=self.config.dev_transform,
            train_empty_mri_ratio = empty_mri_ratio, 
            random_state=self.config.random_state,
            model_name = f'{self.config.model_name}-{self.counter}',
            database_url=self.config.mlflow_database_url,
            checkpoint_dir= self.config.checkpoint_dir
        )

        
        model_trainer(
            num_epochs = self.config.num_epochs,
            accelerator= self.config.accelerator, # defined by test in config entity
            devices = self.config.devices # defined by test in config entity
        )

        logger.logging.info(f'model {self.config.model_name}-{self.counter} successfully trained')

        loss, _ = model_trainer.get_best_model_info()
        self.counter += 1
        
        del model_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        

        return loss

    
    def __call__(self):
        logger.logging.info('Start of optuna opitmization process...')
        self.study.optimize(self._objective, n_trials=self.config.n_trials, show_progress_bar = True)
        self.counter = 0
        logger.logging.info('Optuna optimization successfully finished')
