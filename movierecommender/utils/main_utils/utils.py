import pickle
import os
import yaml
import sys
from movierecommender.exception.exception import MovieRecommenderException
from movierecommender.logging.logger import logging
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MovieRecommenderException(e, sys) from e

def write_yaml_file(file_path: str, content:object, replace: bool = False) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MovieRecommenderException(e, sys) from e
    
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MovieRecommenderException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise  MovieRecommenderException(e, sys) from e
    
def load_numpy_array_data(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ModuleNotFoundError(e, sys) from e

def load_object(file_path: str) -> None:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exists")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise MovieRecommenderException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param, sample_size=100000):
    try:
        report = {}

        # ✅ Step 1: sample data for tuning
        if X_train.shape[0] > sample_size:
            sample_idx = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_train_sample = X_train[sample_idx]
            y_train_sample = y_train[sample_idx]
        else:
            X_train_sample, y_train_sample = X_train, y_train

        for model_name, model in models.items():
            print(f"\n🔍 Tuning {model_name} on {X_train_sample.shape[0]} samples...")

            para = param.get(model_name, {})

            if len(para) > 0:
                # GridSearch only on 100k
                gs = GridSearchCV(model, para, cv=2, n_jobs=-1, verbose=2)
                gs.fit(X_train_sample, y_train_sample)
                best_params = gs.best_params_
                print(f"✅ Best params for {model_name}: {best_params}")
                model.set_params(**best_params)
            else:
                model.fit(X_train_sample, y_train_sample)

            # ✅ Step 2: retrain with best params on full dataset
            print(f"⚡ Retraining {model_name} on full dataset...")
            model.fit(X_train, y_train)

            # predictions
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise MovieRecommenderException(e, sys)
