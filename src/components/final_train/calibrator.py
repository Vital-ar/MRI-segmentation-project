import torch
import os
from torchvision.transforms import v2
from src.components.optuna_model_selection.model import MRIFlexAttentionUNet
from src.components.optuna_model_selection.lightning_modules import MRIDataModule
import numpy as np
import matplotlib.pyplot as plt
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', None) is not None:
    from src.constants.kaggle import TRAIN_CSV
else:
    from src.constants.local import  TRAIN_CSV
from scipy.ndimage import gaussian_filter1d


class ThresholdCalibrator:
    def __init__(self, config, num_integ_bins = 100001, count_shift = True, save_stats = True):

        self.device =torch.device('cuda' if config.accelerator == 'gpu' else 'cpu') 
        state_dict = torch.load(config.inp_model_path, 
                                self.device, weights_only=False)['state_dict']
        inner_state_dict = {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }

        self.model = MRIFlexAttentionUNet( config.inp_channels,
            config.first_conv_out_channels,
            config.num_classes,
            config.depth,
            config.n_encoder_conv_layers,
            config.n_decoder_conv_layers,
            config.kernel_sizes)
        
        
        self.model.load_state_dict(inner_state_dict)

        
        dataset = MRIDataModule(TRAIN_CSV, config.dev_csv, config.test_csv, batch_size = config.batch_size, 
                                num_workers=config.num_workers, train_empty_mri_ratio= 1,
                                train_transform=v2.Compose([
                                    v2.Resize(256),
                                    v2.CenterCrop(256)
                                ]),
                                dev_transform=v2.Compose([
                                    v2.Resize(256),
                                    v2.CenterCrop(256)
                                    ]))
        
        dataset.setup()
        self.train_dataloader = dataset.train_dataloader()
        self.val_dataloader = dataset.val_dataloader()
        self.test_dataloader = dataset.test_dataloader()
        self.device =torch.device('cuda' if config.accelerator == 'gpu' else 'cpu')
        self.num_channels = config.inp_channels
        self.num_integ_bins = num_integ_bins
        self.best_ratio = np.array([0.5,0.5,0.5])
        self.count_shift = count_shift
        self.final_model_path = config.final_checkpoint
        self.save_stats = save_stats
        self.model_name = config.model_name



    def __call__(self):
        state_dict = torch.load(self.final_model_path, 
                                'cpu', weights_only=False)['state_dict']
        inner_state_dict = {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }
        self.model.load_state_dict(inner_state_dict)

        dist_pos_train, dist_neg_train = self.get_distribution(self.train_dataloader)
        dist_pos_val, dist_neg_val = self.get_distribution(self.val_dataloader)
        self.dist_pos_final = dist_pos_train + dist_pos_val
        self.dist_neg_final = dist_neg_train + dist_neg_val

        self.final_best_f1_threshold, final_f1 = self.best_f1_threshold(self.dist_pos_final, self.dist_neg_final)

        
        self.threshold = self.final_best_f1_threshold
        if self.count_shift:
            self.threshold += self.final_shift


        self.dist_pos_test, self.dist_neg_test = self.get_distribution(self.test_dataloader)

        thresh_bin = (self.threshold*(self.num_integ_bins-1)).astype(np.int64)
        fp = np.array([np.sum(self.dist_neg_test[c, thresh_bin[c]:]) for c in range(self.num_channels)])
        fn = np.array([np.sum(self.dist_pos_test[c, :thresh_bin[c]]) for c in range(self.num_channels)])
        tp = np.array([np.sum(self.dist_pos_test[c, thresh_bin[c]:]) for c in range(self.num_channels)])
        eps = 10e-7
        f1 = (2*tp + eps)/ (2*tp+fp+fn+eps)
        recall = (tp + eps) / (tp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        print(f'the final f1 score computed from probs {f1}')

        if self.save_stats:
            os.makedirs(f'stats/{self.model_name}', exist_ok=True)
            np.save(f'stats/{self.model_name}/train_stat.npy', np.array([self.dist_pos_train, self.dist_neg_train]))
            np.save(f'stats/{self.model_name}/final_stat.npy', np.array([self.dist_pos_final, self.dist_neg_final]))
            np.save(f'stats/{self.model_name}/val_stat.npy', np.array([self.dist_pos_val, self.dist_neg_val]))
            np.save(f'stats/{self.model_name}/test_stat.npy', np.array([self.dist_pos_test, self.dist_neg_test]))
            np.save(f'stats/{self.model_name}/f1_optimized_threshold.npy', self.threshold)
            np.save(f'stats/{self.model_name}/f1_recall_precision_metrics.npy', np.array([f1, recall, precision]))



    def calc_best_thresh_and_shift(self):
        
        self.dist_pos_train, self.dist_neg_train = self.get_distribution(self.train_dataloader)
        self.dist_pos_val, self.dist_neg_val = self.get_distribution(self.val_dataloader)
        self.final_pos_val = self.dist_pos_train + self.dist_pos_val
        self.dist_neg_final = self.dist_neg_train + self.dist_neg_val

        self.train_best_f1_threshold, train_f1 = self.best_f1_threshold(self.dist_pos_train, self.dist_neg_train)
        self.val_best_f1_threshold, val_f1 = self.best_f1_threshold(self.dist_pos_val, self.dist_neg_val)
        if self.count_shift:
            shift = self.val_best_f1_threshold - self.train_best_f1_threshold
            self.final_shift = shift  #np.random.randn()*shift + shift

            print(train_f1, val_f1)



    def get_distribution(self, dataloader):

        dist_pos = np.zeros((self.num_channels, self.num_integ_bins), dtype=np.int64)  
        dist_neg = np.zeros((self.num_channels, self.num_integ_bins), dtype=np.int64)  
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inp, mask = batch
                inp = inp.to(self.device)
                mask = mask.to(self.device)
                out = torch.sigmoid(self.model(inp))  

                for c in range(self.num_channels):
                    probs_c = out[:, c].reshape(-1)
                    mask_c = mask[:, c].reshape(-1)

                    pos_vals = probs_c[mask_c == 1]
                    neg_vals = probs_c[mask_c == 0]

                    pos_bins = (pos_vals * (self.num_integ_bins-1)).round().type(torch.int64).cpu().numpy()
                    neg_bins = (neg_vals * (self.num_integ_bins-1)).round().type(torch.int64).cpu().numpy()

                    dist_pos[c] += np.bincount(pos_bins, minlength=self.num_integ_bins)
                    dist_neg[c] += np.bincount(neg_bins, minlength=self.num_integ_bins)
                print(i)
                break
                if i%100 == 0:
                    print(i, 'new batches where added to distribution')
               
        return dist_pos, dist_neg



    def plot_smooth_distributions(self, dist_pos = None, dist_neg = None, sigma=50, left_x_lim = 0, right_x_lim = 1, threshold = None):

        if dist_pos is None:
            dist_pos = np.copy(self.dist_pos_train)

        if dist_neg is None:
            dist_neg = np.copy(self.dist_neg_train)
        x = np.linspace(0, 1, self.num_integ_bins)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        channel_names = ['small bowel', 'large bowel', 'stomach']
        if threshold is None and self.best_threshold is not None:
            threshold = self.best_threshold
        
        for c in range(3):
            ax = axes[c]
            if sigma == 0:
                smooth_pos = dist_pos[c].astype(float)
                smooth_neg = dist_neg[c].astype(float)

            else:
                smooth_pos = gaussian_filter1d(dist_pos[c].astype(float), sigma=sigma)
                smooth_neg = gaussian_filter1d(dist_neg[c].astype(float), sigma=sigma)
            
            ax.plot(x, smooth_pos, label='Foreground', color='red', linewidth=2)
            ax.plot(x, smooth_neg, label='Background', color='blue', linewidth=2)
            if threshold is not None:
                ax.axvline(x = threshold[c], label = 'Threshold', color = 'green', linewidth = 1)
            
            ax.fill_between(x, smooth_pos, alpha=0.2, color='red')
            ax.fill_between(x, smooth_neg, alpha=0.2, color='blue')
            
            ax.set_title(f'{channel_names[c]}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Probability', fontsize=12)
            ax.set_ylabel('Smoothed Pixel Count', fontsize=12)
            
            ax.set_yscale('log')
            ax.set_xlim(left_x_lim, right_x_lim)
            
            ax.set_ylim(bottom=1) 
            
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best')
            
        plt.tight_layout()
        plt.show()




    def get_partial_distribution(self,dist_arr, neg_arr = False, ratio = [1,1,1]):

        ratio = np.array(ratio)
        part_dist = np.copy(dist_arr)
        lengths = np.sum(part_dist, axis=1)
        amounts_to_discard = np.round(lengths * ([1] - ratio))
        
        if neg_arr:
            part_dist = np.flip(part_dist, 1)
        cum_sums = np.cumsum(part_dist, axis=1)
        mask_to_zero = cum_sums <= amounts_to_discard[:, None]
        
        part_dist[mask_to_zero] = 0
        if neg_arr:
            part_dist = np.flip(part_dist, 1)
        return part_dist



    def find_intersection(self,pos_dist, neg_dist, steps = 10, lr = 5, early_stopping = False):
        
        if self.best_ratio is not None:
            ratio = self.best_ratio
        else:
            ratio = np.array([0.5,0.5,0.5])
        best_diff = np.array([float('inf'),float('inf'),float('inf')])
        best_ratio = ratio
        
        best_trial = np.zeros(3, int) 
        
        best_threshold = np.zeros(3)
        for i in range(steps):
            pos_part_dist = self.get_partial_distribution(pos_dist, ratio = ratio)
            neg_part_dist = self.get_partial_distribution(neg_dist, neg_arr = True,ratio = ratio)
            
            #pos_under_dist = np.sum(pos_part_dist != 0 , 1) 
            #neg_under_dist = np.sum(neg_part_dist != 0 , 1)
    
            pos_threshold_idx = np.argmax(pos_part_dist != 0, 1)
            neg_threshold_idx = self.num_integ_bins - 1 - np.argmax(np.flip(neg_part_dist != 0, 1), 1)
    
            #diff = len(pos_dist[0]) - pos_under_dist - neg_under_dist
            diff = pos_threshold_idx - neg_threshold_idx
            diff = np.array(diff)
            abs_diff = abs(diff)
            #optim_threshold = (len(pos_dist[0]) - pos_under_dist - diff//2)/len(pos_dist[0])
            optim_threshold = (pos_threshold_idx + neg_threshold_idx) / 2 / (self.num_integ_bins - 1)
            better = abs_diff < abs(best_diff)

            best_diff[better] = diff[better]
            best_ratio[better] = ratio[better]
            best_trial[better] = i
            best_threshold[better] = optim_threshold[better]
            print('='*20)
            print(*[
                f'''\n----------------------------------------
                \nbest difference for {c}th channel 
                \nfalls under {best_ratio[c]*100}% distribution 
                \nrecorded at {best_trial[c]} 
                \nwith difference of {best_diff[c]} 
                ''' for c in range(len(better))])

            if early_stopping and np.sum(better)==0: 
                break

            ratio+= diff / (self.num_integ_bins - 1)*lr
            ratio = np.clip(ratio, 0.01, 0.99)

            self.best_ratio = best_ratio
        return best_ratio, best_diff, best_trial, best_threshold
         



    def tpr_tnr_crossing(self, dist_pos, dist_neg):
        total_pos = dist_pos.sum(axis=1, keepdims=True)
        total_neg = dist_neg.sum(axis=1, keepdims=True)

        tp = np.cumsum(dist_pos[:, ::-1], axis=1)[:, ::-1]  
        tn = np.cumsum(dist_neg, axis=1)                      

        tpr = tp / total_pos
        tnr = tn / total_neg

        diff = np.abs(tpr - tnr)
        best_bin = np.argmin(diff, axis=1)
        best_threshold = best_bin / (dist_pos.shape[1] - 1)
        return best_threshold



    def best_f1_threshold(self, dist_pos, dist_neg):
        
        tp = np.cumsum(dist_pos[:, ::-1], axis=1)[:, ::-1]
        fp = np.cumsum(dist_neg[:, ::-1], axis=1)[:, ::-1]
        total_pos = dist_pos.sum(axis=1, keepdims=True)
        fn = total_pos - tp

        eps = 10e-7
        f1 = (2*tp + eps) / (2*tp + fp +eps + fn)

        best_bin = np.argmax(f1, axis=1)
        best_f1 = f1[np.arange(self.num_channels), best_bin]
        best_threshold = best_bin / (self.num_integ_bins - 1)

        return best_threshold, best_f1

 




