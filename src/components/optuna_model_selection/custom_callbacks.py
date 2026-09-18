from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
import mlflow
import torch
import os


class MLflowLoggingCallback(Callback):

    
    def __init__(self, labels, dirpath, model_name, random_state = 42):
        
        super().__init__()
        self.model_name= model_name
        self.labels = labels
        self.random_state = random_state
        self.best_f1 = 0
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)

    
    def on_train_start(self, trainer, pl_module):
        
        mlflow.log_param('model_type', 'Attention UNet')
        mlflow.log_param('optimizer', 'AdamW')
        mlflow.log_param('scheduler', 'ReduceLROnPlateau')
        mlflow.log_param('batch_size', trainer.datamodule.batch_size)
        mlflow.log_param('num_workers', trainer.datamodule.num_workers)
        mlflow.log_param('random_state', self.random_state)
        mlflow.log_param('initial_lr', pl_module.hparams.learning_rate)
        mlflow.log_param('weight_decay', pl_module.hparams.weight_decay)
        mlflow.log_param('inp_channels', pl_module.hparams.inp_channels)
        mlflow.log_param('first_conv_out_channels', pl_module.hparams.first_conv_out_channels)
        mlflow.log_param('num_classes', pl_module.hparams.num_classes)
        mlflow.log_param('depth', pl_module.hparams.depth)
        mlflow.log_param('n_encoder_conv_layers', pl_module.hparams.n_encoder_conv_layers)
        mlflow.log_param('n_decoder_conv_layers', pl_module.hparams.n_decoder_conv_layers)
        mlflow.log_param('kernel_sizes', pl_module.hparams.kernel_sizes)
        
    
    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.sanity_checking:
            return
        
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch

        if not trainer.is_global_zero:
            return

        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=current_epoch)
        
        if 'train_loss' in metrics:
            mlflow.log_metric('train_loss', metrics['train_loss'].item(), step=current_epoch)

        if "val_loss" in metrics:
            mlflow.log_metric('val_loss', metrics['val_loss'].item(), step=current_epoch)

        if 'val_recall' in metrics:
            mlflow.log_metric('val_recall', metrics['val_recall'].item(), step=current_epoch)
        
        if 'val_f1_score' in metrics:
            f1_score = metrics['val_f1_score'].item()
            
            mlflow.log_metric('val_f1_score', f1_score , step=current_epoch)
            
            if f1_score > self.best_f1:
                self.best_f1 = f1_score
                
                safe_val_loss = metrics.get('val_loss', torch.tensor(float('inf'))).item()
                
                checkpoint = {
                    'epoch': current_epoch + 1,
                    'model_state_dict': pl_module.state_dict(),
                    'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                    'val_loss': safe_val_loss,
                    'accuracy': f1_score,
                    'random_seed': self.random_state 
                }
                
                checkpoint_path = os.path.join(self.dirpath, f'{self.model_name}_best.pt')
                torch.save(checkpoint, checkpoint_path)
                
                mlflow.log_artifact(checkpoint_path)

    
    def on_train_end(self, trainer, pl_module):
        
        mlflow.log_metric("best_f1_score", self.best_f1)
        
        
        pl_module.eval()
        b, c, h, w = trainer.datamodule.batch_size, pl_module.hparams.inp_channels, 256, 256
        dummy_input = torch.randn(b, c, h, w, device=pl_module.device).cpu().numpy()
        pl_module.to("cpu")
        
        mlflow.pytorch.log_model(
            pytorch_model=pl_module,
            artifact_path=self.model_name,
            input_example=dummy_input
        )
        
        print(f'\nFinished Training. Best f1 score: {self.best_f1:.4f}%')








class BestValLossCallback(pl.Callback):

    def __init__(self):
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        val_loss = trainer.callback_metrics.get("val_loss")
        print('[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]')

        if val_loss is not None:
            val_loss = val_loss.detach().item()
            print('(((((((((((((((((((((((((((((((((())))))))))))))))))))))))))))))))))')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss