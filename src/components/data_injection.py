from dataclasses import dataclass
from datetime import datetime
import os, sys
import numpy as np
import pandas as pd

from src.logging.logger import logging
from src.exception.exception import CustomException


# ===================================================
# 1️⃣ CONFIGURATION
# ===================================================
@dataclass
class DataInjectionConfig:
    """
    Stores all file paths for data ingestion and saving.
    """
    #time_stamp: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    raw_data_path: str = "/home/leksman/Desktop/my git hub work/end_to_end_Real_Estate_Demand_Predictio/raw_datas"
    train_data_path: str = os.path.join("artifacts", "raw_data", "train_data", "train.csv")
    test_data_path: str = os.path.join("artifacts", "raw_data", "test_data", "test.csv")


# ===================================================
# 2️⃣ DATA INJECTION CLASS
# ===================================================
class DataInjection:
    def __init__(self):
        try:
            self.config = DataInjectionConfig()
            logging.info("✅ DataInjectionConfig initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------------------------------
    # Helper: Add prefixes to columns
    # ---------------------------------------------------
    def prefix_columns(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in ["sector", "month"]}
        return df.rename(columns=rename_map)

    # ---------------------------------------------------
    # Main ingestion logic
    # ---------------------------------------------------
    def inject_data(self, raw_data_path):
        try:
            logging.info("🚀 Starting data ingestion...")

            # 1️⃣ Load all datasets
            logging.info("Loading raw CSV files...")
            ci = (
                pd.read_csv(f"{raw_data_path}/train/city_indexes.csv")
                .head(6)
                .fillna(-1)
                .drop(columns=["total_fixed_asset_investment_10k"])
                .pipe(self.prefix_columns, "ci_")
            )
            sp = pd.read_csv(f"{raw_data_path}/train/sector_POI.csv").fillna(-1).pipe(self.prefix_columns, "sp_")
            train_lt = pd.read_csv(f"{raw_data_path}/train/land_transactions.csv").pipe(self.prefix_columns, "lt_")
            train_ltns = pd.read_csv(f"{raw_data_path}/train/land_transactions_nearby_sectors.csv").pipe(self.prefix_columns, "ltns_")
            train_pht = pd.read_csv(f"{raw_data_path}/train/pre_owned_house_transactions.csv").pipe(self.prefix_columns, "pht_")
            train_phtns = pd.read_csv(f"{raw_data_path}/train/pre_owned_house_transactions_nearby_sectors.csv").pipe(self.prefix_columns, "phtns_")
            train_nht = pd.read_csv(f"{raw_data_path}/train/new_house_transactions.csv").pipe(self.prefix_columns, "nht_")
            train_nhtns = pd.read_csv(f"{raw_data_path}/train/new_house_transactions_nearby_sectors.csv").pipe(self.prefix_columns, "nhtns_")

            # 2️⃣ Extract month/sector for test file
            test = pd.read_csv(f"{raw_data_path}/test.csv")
            test[["month", "sector"]] = test["id"].str.split("_", expand=True)

            # 3️⃣ Month conversion map
            MONTH_CODES = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
            }

            # 4️⃣ Create base dataset (sector × month combinations)
            logging.info("Creating base dataset...")
            sectors = pd.DataFrame({"sector": train_nht["sector"].unique().tolist() + ["sector 95"]})
            months = pd.DataFrame({"month": train_nht["month"].unique()})
            data = months.merge(sectors, how="cross")

            # Add date/time features
            data["sector_id"] = data["sector"].str.split(" ").str[1].astype("int16")
            data["year"] = data["month"].str.split("-").str[0].astype("int16")
            data["month_num"] = data["month"].str.split("-").str[1].map(MONTH_CODES).astype("int8")
            data["time"] = ((data["year"] - 2019) * 12 + data["month_num"] - 1).astype("int16")
            data = data.sort_values(["sector_id", "time"])

            # 5️⃣ Merge all features
            logging.info("Merging features...")
            data = (
                data.merge(train_nht, on=["sector", "month"], how="left").fillna(0)
                    .merge(train_nhtns, on=["sector", "month"], how="left").fillna(-1)
                    .merge(train_pht, on=["sector", "month"], how="left").fillna(-1)
                    .merge(train_phtns, on=["sector", "month"], how="left").fillna(-1)
                    .merge(ci.rename(columns={"ci_city_indicator_data_year": "year"}), on="year", how="left").fillna(-1)
                    .merge(sp, on="sector", how="left").fillna(-1)
                    .merge(train_lt, on=["sector", "month"], how="left").fillna(-1)
                    .merge(train_ltns, on=["sector", "month"], how="left").fillna(-1)
            )

            # 6️⃣ Optimize integers
            for col in data.select_dtypes(include=["int64"]).columns:
                c_min, c_max = data[col].min(), data[col].max()
                if c_min == 0 and c_max == 0:
                    data.drop(columns=[col], inplace=True)
                elif np.iinfo(np.int8).min <= c_min <= np.iinfo(np.int8).max:
                    data[col] = data[col].astype("int8")
                elif np.iinfo(np.int16).min <= c_min <= np.iinfo(np.int16).max:
                    data[col] = data[col].astype("int16")

            data.drop(columns=["month", "sector", "year"], inplace=True)

            # 7️⃣ Rolling window features
            logging.info("Creating rolling features...")
            data = data.sort_values(["sector_id", "time"])
            for col in data.columns[3:]:
                for p in [3, 6, 12]:
                    data[f"{col}_mean{p}"] = data.groupby("sector_id")[col].transform(lambda x: x.rolling(p, min_periods=1).mean())
                    data[f"{col}_min{p}"] = data.groupby("sector_id")[col].transform(lambda x: x.rolling(p, min_periods=1).min())
                    data[f"{col}_max{p}"] = data.groupby("sector_id")[col].transform(lambda x: x.rolling(p, min_periods=1).max())

            # 8️⃣ Lag and cyclical features
            lag = 1
            data["label"] = data.groupby("sector_id")["nht_amount_new_house_transactions"].shift(lag)
            data = data[data["label"] != 0]  # Drop label rows with zero

            data["cs"] = np.cos((data["month_num"] - 1) / 6 * np.pi)
            data["sn"] = np.sin((data["month_num"] - 1) / 6 * np.pi)
            data["cs6"] = np.cos((data["month_num"] - 1) / 3 * np.pi)
            data["sn6"] = np.sin((data["month_num"] - 1) / 3 * np.pi)
            data["cs3"] = np.cos((data["month_num"] - 1) / 1.5 * np.pi)
            data["sn3"] = np.sin((data["month_num"] - 1) / 1.5 * np.pi)

            data.drop(columns=["sector_id"], inplace=True)

            # 9️⃣ Train/test split
            N_TEST_MONTHS = 3
            max_time = data["time"].max()
            border = max_time - N_TEST_MONTHS

            train_df = data[data["time"] <= border].dropna(subset=["label"])
            test_df = data[data["time"] > border].dropna(subset=["label"])

            # 10️⃣ Save datasets
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info(f"✅ Data ingestion complete! Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------------------------------
    # Pipeline trigger
    # ---------------------------------------------------
    def initiate_data_injection(self) -> DataInjectionConfig:
        try:
            if os.path.exists(self.config.train_data_path) and os.path.exists(self.config.test_data_path):
                logging.info("✅ Training and testing data already exist. Skipping data injection.")
            else:
                logging.info("🚀 Starting data injection process...")
                self.inject_data(raw_data_path=self.config.raw_data_path)
                logging.info("✅ Data injection completed successfully.")
            return self.config
        except Exception as e:
            raise CustomException(e, sys)