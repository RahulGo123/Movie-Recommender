import os
import sys

from movierecommender.exception.exception import MovieRecommenderException
from movierecommender.logging.logger import logging

from movierecommender.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from movierecommender.entity.config_entity import ModelTrainerConfig


from movierecommender.utils.ml_utils.model.estimator import MovieModel
from movierecommender.utils.main_utils.utils import save_object, load_object
from movierecommender.utils.main_utils.utils import (
    load_numpy_array_data,
    evaluate_models,
)
from movierecommender.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import joblib

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MovieRecommenderException(e, sys)
        
    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1, n_jobs=-1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1, verbose=1),
            "Adaboost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {"criterion": ["gini", "entropy"]},
            "Random Forest": {"n_estimators": [8, 32]},
            "Gradient Boosting": {"learning_rate": [0.1, 0.01], "n_estimators": [8, 32]},
            "Logistic Regression": {},  # no grid search params here
            "Adaboost": {"learning_rate": [0.1, 0.01], "n_estimators": [8, 32]},
        }

        # run evaluation
        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        # get best model
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        # metrics
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(x_test)

        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # load preprocessor
        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

        # create full pipeline object
        movie_model = MovieModel(preprocessor=preprocessor, model=best_model)

        # ensure model directory
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        # save combined model instead of raw sklearn model
        save_object(self.model_trainer_config.trained_model_file_path, obj=movie_model)
        save_object("final_model/model.pkl", movie_model)

        # artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )
            
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise MovieRecommenderException(e, sys)