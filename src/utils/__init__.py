import numpy as np
from pathlib import Path
import os




def _one_class_mask(organ_idx_str, img_shape):
    organ_arr = organ_idx_str.split()
    organ_nums = [int(a) for a in organ_arr]
    organ_lens = organ_nums[1::2]
    organ_ind = organ_nums[::2]
    organ_flattened_mask = np.zeros(img_shape[0]*img_shape[1])
    for ind, leng in zip(organ_ind, organ_lens):
        organ_flattened_mask[ind-1:ind+leng-1] = 1
    return organ_flattened_mask.reshape(img_shape)

def make_img_mask(large_bowel_idx_str, small_bowel_idx_str, stomach_idx_str, img_shape):
    large_bowel_mask = _one_class_mask(large_bowel_idx_str, img_shape)
    small_bowel_mask = _one_class_mask(small_bowel_idx_str, img_shape)
    stomach_mask = _one_class_mask(stomach_idx_str, img_shape)
    img_mask = np.stack([large_bowel_mask, small_bowel_mask, stomach_mask], axis=-1)
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

