from src.pipeline import MakePipe
from src.entity.config_entity import MetaDataEntity
from src.components.form_metadata import FormMetaDataComponent



if __name__ == '__main__':
    meta_pipe = MakePipe('METADATA FORMATION STAGE', MetaDataEntity, FormMetaDataComponent)
    meta_pipe()