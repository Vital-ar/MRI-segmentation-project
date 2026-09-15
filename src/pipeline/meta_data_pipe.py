from src.pipeline import MakePipe
from src.entity.config_entity import MetaDataEntity
from src.components.form_metadata import FormMetaDataComponent



def run_meta_data_pipe():
    meta_pipe = MakePipe('MODEL SELECTION STAGE', MetaDataEntity, FormMetaDataComponent)
    meta_pipe()

if __name__ == '__main__':
    run_meta_data_pipe()