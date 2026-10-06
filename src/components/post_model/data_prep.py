import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.components.optuna_model_selection.model import MRIFlexAttentionUNet
from src.components.optuna_model_selection.dataset import MRISaverDataset
from src.utils import generate_kaggle_csv

class PostModelDataCreation:
    def __init__(self, config):

        self.device =torch.device('cuda' if config.accelerator == 'gpu' else 'cpu')
        ckpt = torch.load(config.base_model_path, 
                                self.device, weights_only=False)
        state_dict = ckpt['state_dict']
        inner_state_dict = {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }

        hparams =ckpt['hyper_parameters']
        first_conv_out_channels = hparams['first_conv_out_channels']
        depth = hparams['depth']
        n_encoder_conv_layers = hparams['n_encoder_conv_layers']
        n_decoder_conv_layers = hparams['n_decoder_conv_layers']
        kernel_sizes = hparams['kernel_sizes']
        self.threshold = torch.from_numpy(np.load(config.threshold_path))

        self.model = MRIFlexAttentionUNet(config.inp_channels,
            first_conv_out_channels,
            config.num_classes,
            depth,
            n_encoder_conv_layers,
            n_decoder_conv_layers,
            kernel_sizes)
        
        self.model.load_state_dict(inner_state_dict)

        self.make_post_model_dfs(config.train_csv, config.dev_csv, config.test_csv, config.detailed_csv)
        print(config.train_csv)
        self.dataset = MRISaverDataset(self.df,transform=config.transform)
        self.dataloader = DataLoader(self.dataset, config.batch_size , False, num_workers= config.num_workers)



    def make_post_model_dfs(self, train_df_path, val_df_path, test_df_path, detailed_df_path):
        
        train_df, self.post_train_df_path = self._make_df(train_df_path, detailed_df_path)
        val_df, self.post_val_df_path = self._make_df(val_df_path, detailed_df_path)
        test_df, self.post_test_df_path = self._make_df(test_df_path, detailed_df_path)
        self.df = pd.concat([train_df, val_df, test_df], ignore_index= True)



    def _make_df(self, df_path: Path, detailed_df_path):

        stem = df_path.stem.removesuffix('_kaggle')
        if df_path.stem.endswith('_kaggle'):
            full_df_path = df_path.parent / (stem + '_post_model' + '_kaggle' + df_path.suffix)
        else:
            full_df_path = df_path.parent / (stem + '_post_model' + df_path.suffix)

        print(full_df_path)
        if full_df_path.is_file():
            full_df = pd.read_csv(full_df_path)


        else:
            df = pd.read_csv(df_path)
            detailed_df = pd.read_csv(detailed_df_path)

            full_df = df.merge(detailed_df, on = ['empty_slice', 'prev_path', 'path', 'next_path'])
            full_df = full_df.drop(['large_bowel', 'small_bowel', 'stomach', 'height', 'width'], axis = 1)
            full_df = full_df[['collapsed_id', 'case', 'slice', 'empty_slice', 'path', 'prev_path', 'next_path', 'mask_path']]
            full_df['mask_path'] = full_df['mask_path'].apply(lambda p: Path(p))


            def make_collapsed_mask_path(d):
                p = d.with_name(re.sub(r'_\d+$', '', d.stem) + '.pt')
                new_path = p.parent.with_name('post_model')/ 'collapsed_mask' / p.name
                return new_path
            
            full_df['collapsed_mask_path'] = full_df['mask_path'].apply(make_collapsed_mask_path)
            


            def make_model_mask_path(d):
                p = d.with_name( d.stem + '.pt')
                new_path = p.parent.with_name('post_model')/ 'model_mask' / p.name
                return new_path

            full_df['model_mask_path'] = full_df['mask_path'].apply(make_model_mask_path)
            


            def make_out_transformed_mask_path(d):
                p = d.with_name(d.stem + '.pt')
                new_path = p.parent.with_name('post_model')/ 'out_transformed_mask' / p.name
                return new_path

            full_df['out_transformed_mask_path'] = full_df['mask_path'].apply(make_out_transformed_mask_path)
            

            full_df.to_csv(full_df_path, index = False)

        return full_df, full_df_path


    def save_model_and_out_mask(self):
        
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                
                inp, idx = batch

                all_exist = np.prod([Path(self.dataset.partial_df.model_mask_path[int(id_)]).is_file() for id_ in idx], dtype = bool)

                if not all_exist:
                    inp = inp.to(self.device)
                    out = self.model(inp)

                    for j in range(len(out)):
                        out_mask_path = Path(self.dataset.partial_df.model_mask_path[int(idx[j])])
                        if not out_mask_path.parent.is_dir():
                            out_mask_path.parent.mkdir(parents=True, exist_ok = True)
                        if not out_mask_path.is_file():
                            torch.save(out[j], out_mask_path)

               

    def save_collapsed_mask(self):
        
        df = self.dataset.partial_df[~self.dataset.partial_df.empty_slice]    
        grooped_obj = [p for p in df.groupby('collapsed_mask_path', sort = False)['model_mask_path']]

        for gr in grooped_obj:
            save = gr[0]
            
            collapsed_mask = []
            model_mask_path  = gr[1].to_numpy()
            
            for p in model_mask_path:
                out = torch.load(p, self.device)
                probs = torch.sigmoid(out) 
                thresh = self.threshold.view(1, -1, 1, 1)
                mask = (probs >= thresh).int()
                collapsed_mask.append(mask)

            collapsed_mask = np.array(collapsed_mask)
            collapsed_mask = torch.from_numpy(collapsed_mask)
            collapsed_mask = torch.sum(collapsed_mask, dim = 0)
            print(collapsed_mask.shape)
            if not save.parent.is_dir():
                save.parent.mkdir(parents=True, exist_ok = True)
            if not save.is_file():
                torch.save(collapsed_mask, save)
            break



    def __call__(self):

        mandat_col_imgs = self.dataset.partial_df.collapsed_mask_path.unique()
        collapsed_mask_dir = Path(self.dataset.partial_df.collapsed_mask_path[0]).parent
        if collapsed_mask_dir.is_dir():
            num_col_imgs = sum(1 for a in collapsed_mask_dir.iterdir() if a in mandat_col_imgs)
        else: 
            num_col_imgs = 0


        model_mask_dir = Path(self.dataset.partial_df.model_mask_path[0]).parent
        mandat_model_imgs = self.dataset.partial_df.model_mask_path.to_numpy()
        if model_mask_dir.is_dir():
            num_imgs = sum(1 for a in model_mask_dir.iterdir() if a in mandat_model_imgs)
        else:
            num_imgs = 0

        if len(mandat_col_imgs) > num_col_imgs:

            if len(self.dataset) > num_imgs:
                self.save_model_and_out_mask()

            self.save_collapsed_mask()

        if (num_col_imgs == len(mandat_col_imgs)) and (len(self.dataset) > num_imgs) :
            print('all is perfectly saved')


    def _create_kaggle_dfs_locally_for_data_prep(self):
        generate_kaggle_csv(self.post_train_df_path, ['path','prev_path','next_path','mask_path'], '/kaggle/input/datasets/vitaliilavryk/mri-segmentation-dataset',['collapsed_mask_path','model_mask_path','out_transformed_mask_path'])
        generate_kaggle_csv(self.post_val_df_path, ['path','prev_path','next_path','mask_path'], '/kaggle/input/datasets/vitaliilavryk/mri-segmentation-dataset', ['collapsed_mask_path','model_mask_path','out_transformed_mask_path'])
        generate_kaggle_csv(self.post_test_df_path, ['path','prev_path','next_path','mask_path'], '/kaggle/input/datasets/vitaliilavryk/mri-segmentation-dataset', ['collapsed_mask_path','model_mask_path','out_transformed_mask_path'])
        
