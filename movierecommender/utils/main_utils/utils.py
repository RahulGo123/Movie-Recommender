<<<<<<< HEAD
import pickle
=======
>>>>>>> 52f92af1defe72e05ea1330c53b1e5ca4531ba01
import os
import yaml
import sys
from movierecommender.exception.exception import MovieRecommenderException
<<<<<<< HEAD
from movierecommender.logging.logger import logging
import numpy as np
=======
>>>>>>> 52f92af1defe72e05ea1330c53b1e5ca4531ba01

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
<<<<<<< HEAD
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
    
=======
        raise MovieRecommenderException(e, sys) from e
>>>>>>> 52f92af1defe72e05ea1330c53b1e5ca4531ba01
