from pathlib import Path

DATA_DIR = Path('data')
ZIPPED_DATA_FILE = Path(DATA_DIR, 'uw-madison-gi-tract-image-segmentation.zip')
EXTRACTED_DATA_DIR = Path(DATA_DIR, 'extracted_data')
PREPROCESSED_DATA_DIR = Path(DATA_DIR, 'preprocessed_data')
METADATA_NAME = Path(PREPROCESSED_DATA_DIR, 'final_dataframe.csv')
IMAGES_DIR = Path(EXTRACTED_DATA_DIR, 'train')
