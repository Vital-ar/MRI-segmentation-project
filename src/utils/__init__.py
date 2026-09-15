import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import cv2
import torch

def _one_class_mask(organ_idx_str, height, width):
    if type(organ_idx_str) == str:
        organ_arr = organ_idx_str.split()
        organ_nums = [int(a) for a in organ_arr]
        organ_lens = organ_nums[1::2]
        organ_ind = organ_nums[::2]
        organ_flattened_mask = np.zeros(height*width, dtype = np.bool)
        for ind, leng in zip(organ_ind, organ_lens):
            organ_flattened_mask[ind-1:ind+leng-1] = 1
        return organ_flattened_mask.reshape((height, width))
    else:
        return np.zeros((height, width), dtype = np.bool)

def make_img_mask(large_bowel_idx_str, small_bowel_idx_str, stomach_idx_str, height, width):
    large_bowel_mask = _one_class_mask(large_bowel_idx_str, height, width)
    small_bowel_mask = _one_class_mask(small_bowel_idx_str, height, width)
    stomach_mask = _one_class_mask(stomach_idx_str, height, width)
    img_mask = np.stack([large_bowel_mask, small_bowel_mask, stomach_mask])
    return img_mask

def _construct_id_from_path(img_path:Path):
    parts = img_path.parts
    case_and_day = parts[4]
    slice = parts[-1].split('_')[1]
    img_id = f'{case_and_day}_slice_{slice}'
    return img_id

def _generate_file_list(dir_path):
    try:
        for item in Path(dir_path).iterdir():
            if item.is_dir():
                # yield from recursively yields items from the sub-generator
                yield from _generate_file_list(item)
            elif item.is_file():
                yield item
                
    except PermissionError:
        print(f"Permission denied: '{dir_path}'")
    except Exception as e:
        print(f"Error: {e}")


def get_path_id_dict(dir_path):
    file_id_dict = {'path': [], 'id': []}
    try:
        for file in _generate_file_list(dir_path):
            file_id_dict['path'].append(file)
            file_id_dict['id'].append(_construct_id_from_path(file))
        return file_id_dict
    except Exception as e:      
        return None

def show_mask_img(orig_img_norm_arr, mask_arr, alpha = 0.5):

    img_norm = orig_img_norm_arr.astype(float)
    if np.max(img_norm) > 1.0:
        img_norm = img_norm / 255.0

    # Create a 3-channel RGB image
    colour_img = np.stack([img_norm, img_norm, img_norm], axis=-1)
    colour_img[mask_arr[0] == 1] = colour_img[mask_arr[0] == 1] * (1 - alpha) + np.array([1, 0, 0]) * alpha
    colour_img[mask_arr[1] == 1] = colour_img[mask_arr[1] == 1] * (1 - alpha) + np.array([0, 1, 0]) * alpha 
    colour_img[mask_arr[2] == 1] = colour_img[mask_arr[2] == 1] * (1 - alpha) + np.array([0, 0, 1]) * alpha

    plt.subplot(1, 2, 1)
    plt.imshow(colour_img)
    plt.title("masked mri slice")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(orig_img_norm_arr, cmap = 'gray')
    plt.title("Normalized MRI Slice")
    plt.axis('off')



def img_norm_16_to_8(img):
    '''
    Args:
        img_path: array like 16-bit image

    Return:
        img_8: numpy.ndarray object of 8-bit image
    '''
    img_norm = cv2.normalize(img, None, alpha=0, beta = 255, norm_type=cv2.NORM_MINMAX)
    img_8 = img_norm.astype(np.uint8)
    return img_8



def show_mask_img_from_tensor(orig_img_tensor, mask_tensor, alpha=0.5):

    img = orig_img_tensor.detach().cpu().float()
    mask = mask_tensor.detach().cpu().float()
    
    if img.ndim == 4:
        img = img[0]
        mask = mask[0]
        
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0) 
        
    # FIX: Remove the 'if' condition. 
    # Force dynamic normalization to [0, 1] every single time.
    # This fixes the black left image and restores contrast to the right image.
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)

    colour_img = torch.stack([img, img, img], dim=-1)
    
    colors = [
        torch.tensor([1., 0., 0.]), 
        torch.tensor([0., 1., 0.]), 
        torch.tensor([0., 0., 1.])  
    ]
    
    for i in range(3):
        m = mask[i] >= 0.5 
        colour_img[m] = colour_img[m] * (1 - alpha) + colors[i] * alpha

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(colour_img.numpy())
    plt.title("Masked MRI Slice")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # By passing vmin=0 and vmax=1, we stop Matplotlib from auto-scaling 
    # the 2D array, ensuring both sides match perfectly.
    plt.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title("Normalized MRI Slice")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()







