from dataclasses import dataclass
#from datetime import datetime
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.logging.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import save_numpy_array_data, save_object


@dataclass
class DataTransformationConfig:
    raw_train_data_path:  str = os.path.join("artifacts", "raw_data", "train_data", "train.csv")
    raw_test_data_path: str = os.path.join("artifacts",  "raw_data","test_data", "test.csv")

    #time_stamp: str = f"D_T_At_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    transformed_train_data_path: str = os.path.join("artifacts", "transformed_dataset", "train.npy")
    transformed_test_data_path: str = os.path.join("artifacts", "transformed_dataset", "test.npy")
    processor_model_path: str = os.path.join("final_model", "process_model.pkl")


class DataTransformation:
    def __init__(self):
        try:
            self.data_trans_config = DataTransformationConfig()
        except Exception as e:
            raise CustomException(e, sys)

    def transform_data(self, raw_train_data_path, raw_test_data_path):
        try:
            logging.info("Loading raw train and test data...")
            train_df = pd.read_csv(raw_train_data_path)
            test_df = pd.read_csv(raw_test_data_path)

            logging.info("Dropping rows where target (nht_amount_new_house_transactions) is zero...")
            train_df = train_df[train_df["nht_amount_new_house_transactions"] != 0]
            test_df = test_df[test_df["nht_amount_new_house_transactions"] != 0]

            logging.info("Splitting into features and target...")
            X_train = train_df.drop("nht_amount_new_house_transactions", axis=1)
            X_test = test_df.drop("nht_amount_new_house_transactions", axis=1)
            y_train = train_df["nht_amount_new_house_transactions"]
            y_test = test_df["nht_amount_new_house_transactions"]

            logging.info("Applying StandardScaler to scale features...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # only transform on test

            logging.info("Combining scaled features with target variable...")
            processed_train = np.c_[X_train_scaled, np.array(y_train).reshape(-1, 1)]
            processed_test = np.c_[X_test_scaled, np.array(y_test).reshape(-1, 1)]

            logging.info("Saving transformed datasets and scaler model...")
            save_numpy_array_data(
                file_path=self.data_trans_config.transformed_train_data_path, array=processed_train
            )
            save_numpy_array_data(
                file_path=self.data_trans_config.transformed_test_data_path, array=processed_test
            )
            save_object(
                file_path=self.data_trans_config.processor_model_path, obj=scaler
            )

            logging.info("Data transformation completed successfully ✅")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transform_data(self):
        self.transform_data(
            self.data_trans_config.raw_train_data_path,
            self.data_trans_config.raw_test_data_path
        )









        



