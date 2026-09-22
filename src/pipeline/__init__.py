from src.logging import logger



class MakePipe:
    def __init__(self, stage_name, entity, component, model_index = None):
        self.stage_name = stage_name
        self.entity = entity
        self.component = component
        self.model_index = model_index

    def __call__(self, **kwargs):
        try:
            logger.logging.info(f'\n----------------stage [{self.stage_name}] started--------------------------')

            if self.model_index is not None:
                comp = self.component(self.entity(self.model_index))

            else:
                comp = self.component(self.entity())
            comp(**kwargs)

            logger.logging.info(f'================stage [{self.stage_name}] ended===========================\n')
        except Exception as e:
            logger.logging.error(f'SOME TROUBLES WITH STAGE [{self.stage_name}]')
            raise e
