import numpy as np
import os,sys
import pickle
from src.logging.logger import logging
from src.exception.exception import CustomException
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise CustomException(e,sys)

import os, pickle, logging

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of mainUtils class")
        
        # make sure the parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # save the object into the file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Exited the save_object method of mainUtils class")

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path:str,) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            print(file_path)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def load_numpy_array_data(file_path:str) -> np.array:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            print(file_path)
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(y_true,y_pred):
    try:
        mse = mean_squared_error(y_true,y_pred)
        rmse = np.sqrt(mean_squared_error(y_true,y_pred))
        mae = mean_absolute_error(y_true,y_pred)
        r2_scores = r2_score(y_true,y_pred)

        return {
            "mse":mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score":r2_scores
        }
    except Exception as e:
        raise CustomException(e,sys)

