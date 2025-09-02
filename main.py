from movierecommender.components.data_ingestion import DataIngestion
from movierecommender.components.data_validation import DataValidation
<<<<<<< HEAD
from movierecommender.components.data_transformation import DataTransformation
from movierecommender.exception.exception import MovieRecommenderException
from movierecommender.logging.logger import logging

from movierecommender.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig
=======
from movierecommender.exception.exception import MovieRecommenderException
from movierecommender.logging.logger import logging

from movierecommender.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig
>>>>>>> 52f92af1defe72e05ea1330c53b1e5ca4531ba01

import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        
<<<<<<< HEAD
        # DATA INGESITON
=======
>>>>>>> 52f92af1defe72e05ea1330c53b1e5ca4531ba01
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        
<<<<<<< HEAD
        # DATA VALIDATION
=======
>>>>>>> 52f92af1defe72e05ea1330c53b1e5ca4531ba01
        datavalidationconfig = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, datavalidationconfig)
        logging.info("Initiate Data Validation")
        datavalidationartifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Complete")
        print(datavalidationartifact)
        
<<<<<<< HEAD
        # DATA TRANSFORMATION
        datatransformationconfig = DataTransformationConfig(trainingpipelineconfig)
        logging.info("data Transformation started")
        data_transformation = DataTransformation(datavalidationartifact, datatransformationconfig)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data Transformation completed")
=======
>>>>>>> 52f92af1defe72e05ea1330c53b1e5ca4531ba01
        
    except Exception as e:
        raise MovieRecommenderException(e,sys)