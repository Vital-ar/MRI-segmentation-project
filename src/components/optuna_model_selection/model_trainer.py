import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path


from src.components.optuna_model_selection.lightning_modules import MRIDataModule, MRIModule
from src.logging import logger






class ModelTrainer:


    def __init__(self,
                 learning_rate = 0.001,
                 weight_decay = 0.01,
                 inp_channels = 3,
                 first_conv_out_channels = 64,
                 num_classes = 3,
                 depth = 3,
                 n_encoder_conv_layers = 2,
                 n_decoder_conv_layers = 2,
                 kernel_sizes: list | int = 3,

                 train_csv = None,
                 dev_csv = None,
                 test_csv = None,
                 batch_size = 32,
                 num_workers = 2,
                 train_empty_mri_ratio = 0.2,
                 train_transform = None,
                 dev_transform = None,
                 
                 random_state = 42, 
                 model_name = 'unet',
                 database_url="sqlite:///mlflow.db",#!
                 checkpoint_dir = '/models/checkpoints'):

        self.model = MRIModule(learning_rate, 
                               weight_decay, 
                               inp_channels, 
                               first_conv_out_channels, 
                               num_classes, 
                               depth, 
                               n_encoder_conv_layers, 
                               n_decoder_conv_layers, 
                               kernel_sizes)

        self.data_module = MRIDataModule(train_csv, 
                                         dev_csv, 
                                         test_csv,
                                         batch_size,
                                         train_transform, 
                                         dev_transform, 
                                         num_workers, 
                                         train_empty_mri_ratio, 
                                         random_state)

        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.callbacks = None
        self.logger = None
        self.database_url = database_url
        self._get_def_callbacks()
        self._get_def_loggers()

        logger.logging.info(f'MRI model trainer successfully initialized')


    def _get_def_callbacks(self):
        dirpath = Path(self.checkpoint_dir, self.model_name)

        checkpoint_callback = ModelCheckpoint(
            dirpath = dirpath, 
            filename ='model-{epoch:02d}-{val_f1_score:.2f}',
            monitor = 'val_f1_score', 
            mode = 'max',    
            verbose = True,           
            save_last = True,
            every_n_epochs = 1
        )


        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=7,
            mode='min',
            verbose=True
        )

        self.callbacks = [checkpoint_callback, early_stop_callback]


    def _get_def_loggers(self):

            self.logger = MLFlowLogger(
                experiment_name="MRI_Segmentation",
                tracking_uri=self.database_url, 
                run_name=self.model_name,
                log_model=True 
            )


    
    def __call__(
            self,
            num_epochs = 50,
            accelerator = 'auto',
            devices = 1
        ):

        self.trainer = pl.Trainer(#max_epochs=1,limit_train_batches=1, limit_val_batches=1, #!delete for real run
             accelerator = accelerator, 
             devices = devices,
             strategy='ddp_spawn',
             logger = self.logger,
             callbacks = self.callbacks,  
             max_epochs = num_epochs, #* uncoment 
             enable_progress_bar = True,
             enable_model_summary = True)
        logger.logging.info(f'MRI model lightning trainer successfully initialized')
        self.trainer.fit(self.model, self.data_module)


    def get_best_model_info(self):

        f1_score =  self.trainer.checkpoint_callback.best_model_score
        path = self.trainer.checkpoint_callback.best_model_path
        if f1_score is None:
            raise ValueError(f"No checkpoint score recorded for monitored metric. Ensure the validation loop completed.")
        return f1_score.item(), path 

