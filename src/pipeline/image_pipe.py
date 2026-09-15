from src.pipeline import MakePipe
from src.entity.config_entity import ImagePrepEntity
from src.components.prepare_images import ImagePreparation


def run_image_preparation_pipe():
    image_pipe = MakePipe('IMAGE PREPARATION STAGE', ImagePrepEntity, ImagePreparation)
    image_pipe(dev_ratio = 0.05, test_ratio = 0.05)


if __name__ == '__main__':
    run_image_preparation_pipe()
