import os, sys
import mlflow
import mlflow.sklearn
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from src.logging.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object, load_numpy_array_data, evaluate_model
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    processed_train_data_path: str = os.path.join("artifacts", "transformed_dataset", "train.npy")
    processed_test_data_path: str = os.path.join("artifacts", "transformed_dataset", "test.npy")
    trained_model_file_path: str = os.path.join("final_model", "model.pkl")


class ModelTrainer:
    def __init__(self):
        try:
            self.model_trainer_config = ModelTrainerConfig()
        except Exception as e:
            raise CustomException(e, sys)

    def mlflow_tracking(self, model_name, model, train_metrics, test_metrics, register_model=True):
        """Logs model, parameters, and metrics to MLflow"""
        try:
            logging.info(f"Starting MLflow tracking for {model_name}")

            with mlflow.start_run(run_name=model_name):
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name if register_model else None,
                )

                # Log parameters (if available)
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                # Log metrics
                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value)
                for key, value in test_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)

            logging.info(f"MLflow tracking completed for {model_name}")
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            # Define models (no fine-tuning)
            models = {
                "RandomForest": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42, verbosity=0),
                "LightGBM": LGBMRegressor(random_state=42),
                "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
                "LinearRegression": LinearRegression()
            }

            for name, model in models.items():
                logging.info(f"Training model: {name}")

                model.fit(x_train, y_train)
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_metrics = evaluate_model(y_train, y_train_pred)
                test_metrics = evaluate_model(y_test, y_test_pred)

                self.mlflow_tracking(
                    model_name=name,
                    model=model,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics
                )

                # Save the last trained model
                save_object(self.model_trainer_config.trained_model_file_path, model)

                logging.info(f"{name} training completed successfully.\n")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_train_model(self):
        try:
            logging.info("Starting initiate_model_trainer")

            # Load processed data
            train_file_path = self.model_trainer_config.processed_train_data_path
            test_file_path = self.model_trainer_config.processed_test_data_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Split features and target
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Train all models
            self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)
