from src.components.post_model.correction_model_training_component import CorrectionModelTrainingComponent
from src.entity.config_entity import CorrectionModelEntity
from src.pipeline import MakePipe


def post_model_train():
    post = MakePipe('POST MODEL TRAINING', CorrectionModelEntity, CorrectionModelTrainingComponent)
    post()

if __name__ == '__main__':
    post_model_train()
