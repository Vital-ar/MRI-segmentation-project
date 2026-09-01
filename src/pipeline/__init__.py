from src.logging import logger



class MakePipe:
    def __init__(self, stage_name, entity, component):
        self.stage_name = stage_name
        self.entity = entity
        self.component = component

    def __call__(self, **kwargs):
        try:
            logger.logging.info(f'----------------stage [{self.stage_name}] started--------------------------')

            comp = self.component(self.entity())
            comp(kwargs)

            logger.logging.info(f'================stage [{self.stage_name}] ended===========================\n')
        except Exception as e:
            logger.logging.error(f'SOME TROUBLES WITH STAGE [{self.stage_name}]')
            raise e
