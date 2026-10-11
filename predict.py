from src.pipeline.mrisegmentator import MRISegmentator
from pathlib import Path
from src.entity.config_entity import MRISegmentationEntity


def _yes_no(message):
    var = None
    while var is None:
        temp_var = input(message).lower()
        if temp_var == 'y':
            var = True
        elif temp_var == 'n':
            var = False

        else:
            print(f'Invalid syntax: {temp_var}. Try again')
def main():
    use_correction_model = _yes_no('Do you want to use correction model on top of base model? \n(it leeds for longer inference but slightly better results) (y,n):')
    segmentator_class = MRISegmentator(MRISegmentationEntity, use_correction_model=use_correction_model)
    more_dir = True
    index = 0
    segmentators = {}
    while more_dir:
        mri_dir = None
        while mri_dir is None:
            mri_dir = Path(input('Pass the path to directory which contains MRI image:\n'))
            if not mri_dir.is_dir():
                print(f'There is no such directory: {mri_dir}, try again')
                mri_dir = None

        save_out = _yes_no('Do you want to save output masks? (y/n)')
       
        save_mask_path = None
        if save_out:
            path_input = input('Enter custom path to save mask (leave empty for default): ')
            if path_input.strip():
                save_mask_path = Path(path_input.strip())

        visualization = _yes_no('Do you want to run visualizations during segmentation? (y/n): ')

        segmentators[index] = segmentator_class(mri_dir, save_out, save_mask_path, visualization)
        index+=1

        more_dir = _yes_no('Do you want to insert more MRI folders masks? (y/n)')


    interactive = _yes_no('Do you want to investigate your MRIs interactively? (y/n):')
    while interactive:
        print(f"\nAvailable segmented MRI indices: {list(segmentators.keys())}")
        chosen_index_str = input("Choose an index to investigate (or 'q' to quit): ")
        
        if chosen_index_str.lower() == 'q':
            break
            
        try:
            chosen_index = int(chosen_index_str)
            if chosen_index not in segmentators:
                print("Index not found in the dictionary. Try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue
            
        current_segmentator = segmentators[chosen_index]
        
        print("\nAvailable functions for investigation:")
        print("1: visualize_mask_3d")
        print("2: visualize_mri_overlay")
        func_choice = input("Choose a function by number (1 or 2): ")
        
        if func_choice == '1':
            current_segmentator.visualize_mask_3d()
        elif func_choice == '2':
            current_segmentator.visualize_mri_overlay()
        else:
            print("Invalid function choice. Please enter 1 or 2.")


