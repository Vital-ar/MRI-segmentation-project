from src.pipeline import MakePipe
from src.entity.config_entity import SelectedModelsCreationEntity
from src.components.v2_training_search import V2TtrainingComponent



def model_search_v2(index):
    v2 = MakePipe('MODEL SELECTION STAGE', SelectedModelsCreationEntity, V2TtrainingComponent, index)
    v2()

if __name__ == '__main__':
    model_search_v2()