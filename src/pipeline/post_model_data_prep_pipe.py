from src.pipeline import MakePipe
from src.entity.config_entity import PostModelDataPrepEntity
from src.components.post_model.data_prep import PostModelDataCreation



def post_model_data_prep():
    post = MakePipe('POST MODEL DATA PREPARATION', PostModelDataPrepEntity, PostModelDataCreation)
    post()

if __name__ == '__main__':
    post_model_data_prep()