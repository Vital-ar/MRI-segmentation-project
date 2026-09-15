from src.pipeline.meta_data_pipe import run_meta_data_pipe
from src.pipeline.image_pipe import run_image_preparation_pipe
from src.pipeline.model_selction_pipe import run_model_selection_pipe

from src.constants import RANDOM_STATE
import lightning.pytorch as pl

pl.seed_everything(RANDOM_STATE, workers=True)


if __name__ == '__main__':

    run_model_selection_pipe()