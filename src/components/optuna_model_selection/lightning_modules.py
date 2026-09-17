import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchmetrics import F1Score, Recall     

from src.components.optuna_model_selection.dataset import MRIDataset
from src.components.optuna_model_selection.model import MRIFlexAttentionUNet
from src.logging import logger
from src.components.optuna_model_selection.dice_bce_loss import DiceBCELoss






class MRIDataModule(pl.LightningDataModule):


    def __init__(self, 
                 train_csv,
                 dev_csv,
                 test_csv,
                 batch_size = 32, 
                 train_transform = None,
                 dev_transform = None,
                 num_workers = 2, 
                 train_empty_mri_ratio = 0.2,
                 random_state = 42):
        super().__init__()


        self.train_csv = train_csv
        self.dev_csv = dev_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_empty_mri_ratio = train_empty_mri_ratio
        self.random_state = random_state


        self.dev_transform = dev_transform
        self.train_transform = train_transform

        logger.logging.info(f'MRI lightning data module successfully initialized')


    def setup(self, stage = None):
        
        if stage == 'fit' or stage is None:
            self.train_ds = MRIDataset(self.train_csv, self.train_empty_mri_ratio, self.train_transform, self.random_state)
            self.dev_ds = MRIDataset(self.dev_csv, 1.0, self.dev_transform, self.random_state)

        if stage == 'test' or stage is None:
            self.test_ds = MRIDataset(self.test_csv, 1.0, self.dev_transform, self.random_state)
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, shuffle = True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dev_ds, self.batch_size, shuffle = False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, shuffle = False, num_workers=self.num_workers)










class MRIModule(pl.LightningModule):


    def __init__(self, 
                learning_rate: float = 0.001,
                weight_decay: float = 0.01, 
                inp_channels: int = 3, 
                first_conv_out_channels: int = 64, 
                num_classes: int = 3, 
                depth: int = 3, 
                n_encoder_conv_layers: int = 2, 
                n_decoder_conv_layers: int = 2, 
                kernel_sizes: list | int = 3):
        
        super().__init__()

        self.save_hyperparameters()

        self.model = MRIFlexAttentionUNet(inp_channels, 
                                          first_conv_out_channels, 
                                          num_classes, 
                                          depth, 
                                          n_encoder_conv_layers, 
                                          n_decoder_conv_layers, 
                                          kernel_sizes)

        self.loss_fn = DiceBCELoss()

        self.f1score = F1Score('multilabel', num_labels=3, average='macro')
        self.recall = Recall('multilabel', num_labels = 3, average = 'macro')

        logger.logging.info(f'MRI lightning module successfully initialized')



    def forward(self, x):

        return self.model(x)



    def training_step(self, batch, batch_idx = None):

        images, masks = batch
        logits = self(images)

        loss = self.loss_fn(logits, masks)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        return loss



    def validation_step(self, batch, batch_idx = None):

        images, masks = batch
        logits = self(images)

        loss = self.loss_fn(logits, masks)

        probs = torch.sigmoid(logits)

        f1_score = self.f1score(probs, masks)
        recall = self.recall(probs, masks)

        self.log('val_loss', loss, prog_bar = True, on_epoch=True, sync_dist=True)
        self.log('val_f1_score', f1_score, prog_bar = True, on_epoch=True, sync_dist=True)
        self.log('val_recall', recall, sync_dist=True)



    def test_step(self, batch, batch_idx = None):
    
            images, masks = batch
            logits = self(images)
    
            loss = self.loss_fn(logits, masks)
    
            probs = torch.sigmoid(logits)
    
            f1_score = self.f1score(probs, masks)
            recall = self.recall(probs, masks)
    
            self.log('test_loss', loss, sync_dist=True)
            self.log('test_f1_score', f1_score, sync_dist=True)
            self.log('test_recall', recall, sync_dist=True)



    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr = self.hparams.learning_rate, weight_decay = self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.1, patience = 2)#! 5

        return {
            "optimizer": optimizer,
            "lr_scheduler": {        
        "scheduler": scheduler,
        "interval": "epoch",
        "frequency": 1,
        "monitor": "val_f1_score",
        "strict": True,
        "name": None,
    },
        }



    def on_train_epoch_end(self):

        if self.trainer.datamodule:
            self.trainer.datamodule.train_ds.on_epoch_end()



    def predict_step(self, batch, batch_idx = None):

        images = batch[0]
        logits = self(images)
        prob = torch.sigmoid(logits)

        masks = (prob > 0.5).int()

        return masks



    def get_img_masks(self, image):

        self.eval()
        with torch.no_grad():

            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.to(self.device)

            logits = self(image)
            prob = torch.sigmoid(logits)
            
            return (prob > 0.5).int()
