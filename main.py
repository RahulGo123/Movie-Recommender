from movierecommender.components.data_ingestion import DataIngestion
from movierecommender.components.data_validation import DataValidation
from movierecommender.components.data_transformation import DataTransformation
from movierecommender.components.model_trainer import ModelTrainer

from movierecommender.exception.exception import MovieRecommenderException
from movierecommender.logging.logger import logging

from movierecommender.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from movierecommender.exception.exception import MovieRecommenderException
from movierecommender.logging.logger import logging

import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        # DATA INGESITON
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        
        # DATA VALIDATION
        datavalidationconfig = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, datavalidationconfig)
        logging.info("Initiate Data Validation")
        datavalidationartifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Complete")
        print(datavalidationartifact)
        
        # DATA TRANSFORMATION
        datatransformationconfig = DataTransformationConfig(trainingpipelineconfig)
        logging.info("data Transformation started")
        data_transformation = DataTransformation(datavalidationartifact, datatransformationconfig)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data Transformation completed")
        
        logging.info("Model Training stared")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        logging.info("Model Trained Artifact created")

    except Exception as e:
        raise MovieRecommenderException(e,sys)