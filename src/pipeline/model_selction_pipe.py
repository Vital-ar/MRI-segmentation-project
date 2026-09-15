from src.pipeline import MakePipe
from src.entity.config_entity import ModelCreationEntity
from src.components.optuna_model_selection.optuna_study import OptunaModelSelectionComponent

def run_model_selection_pipe():
    model_pipe = MakePipe('METADATA FORMATION STAGE', ModelCreationEntity, OptunaModelSelectionComponent)
    model_pipe()


if __name__ == '__main__':
    run_model_selection_pipe()
