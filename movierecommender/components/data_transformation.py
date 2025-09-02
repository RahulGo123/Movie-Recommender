from scipy import sparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from movierecommender.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)
from movierecommender.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from movierecommender.entity.config_entity import DataTransformationConfig
from movierecommender.exception.exception import MovieRecommenderException
from movierecommender.logging.logger import logging
from movierecommender.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            
        except Exception as e:
            raise MovieRecommenderException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MovieRecommenderException(e, sys)
            
    @classmethod
    def get_data_transformer_object(cls) ->Pipeline:
        logging.info("Entered get_data_transformer_oject method of transformation class")
        try:
            numerical_cols = ["timestamp"]
                
            categorical_cols = ["genres"]
                
            num_pipeline = Pipeline([
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ("scaler", StandardScaler())
            ])
                
            cat_pipeline = Pipeline(steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
                
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )
                
            logging.info("Created preprocessin pipeline")
            return preprocessor
        except Exception as e:
            raise ModuleNotFoundError(e, sys)
            
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]


            # Get pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit on training and transform both
            logging.info(f"Train input feature df shape: {input_feature_train_df.shape}")
            logging.info(f"Test input feature df shape: {input_feature_test_df.shape}")

            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)

            # Debug shapes
            logging.info(f"Train features shape: {transformed_input_train_feature.shape}")
            logging.info(f"Train target shape: {target_feature_train_df.shape}")
            logging.info(f"Test features shape: {transformed_input_test_feature.shape}")
            logging.info(f"Test target shape: {target_feature_test_df.shape}")


            # Convert to dense numpy arrays
            if sparse.issparse(transformed_input_train_feature):
                transformed_input_train_feature = transformed_input_train_feature.toarray()
            if sparse.issparse(transformed_input_test_feature):
                transformed_input_test_feature = transformed_input_test_feature.toarray()

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df).reshape(-1, 1)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df).reshape(-1, 1)]

            # Save transformed dataaaaaaaa
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            # Final artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise MovieRecommenderException(e, sys)
