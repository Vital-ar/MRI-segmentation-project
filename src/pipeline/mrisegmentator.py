from src.components.MRI_unet_with_calibrated_thresholds_and_correction import MRISegmentationModel
from src.entity.config_entity import MRISegmentationEntity
from pathlib import Path
import numpy as np

class MRISegmentator:
    def __init__(self):
        self.segmentator = MRISegmentationModel(MRISegmentationEntity)
        print('Segmentatator successfully initialized')

    def __call__(self, mri_dir: Path,save_masks = False, save_mask_path: Path = None, visualization = False):
        self.segmentator(mri_dir)
        if visualization:
            self.segmentator.visualize_mask_3d()
            self.segmentator.visualize_mri_overlay()
        if save_masks:
            if save_mask_path is None:
                save_masks_path = mri_dir.parent / (mri_dir.name + '_masks.npz')

            if not save_masks_path.parent.is_dir():
                save_mask_path.parent.mkdir(parents = True, exist_ok=True)
            np.savez_compressed(save_mask_path, self.segmentator.final_masks.numpy())
        return self.segmentator