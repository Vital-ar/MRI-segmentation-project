from src.entity.config_entity import FinalModelCreationEntity
from src.components.v2_training_search import V2TtrainingComponent
from src.components.final_train.calibrator import ThresholdCalibrator
from src.logging import logger


def final_train():
    stage_name = 'FINAL MODEL TRAIN STAGE'
    try:
        logger.logging.info(f'\n----------------stage [{stage_name}] started--------------------------')

        entity = FinalModelCreationEntity()
        
        trainer = V2TtrainingComponent(entity)


        
        trainer()
        calibrator = ThresholdCalibrator(entity)
        calibrator.calc_best_thresh_and_shift()
        calibrator()

        logger.logging.info(f'================stage [{stage_name}] ended===========================\n')
    except Exception as e:
        logger.logging.error(f'SOME TROUBLES WITH STAGE [{stage_name}]')
        raise e

if __name__ == '__main__':
    final_train()