# 🏠 Real Estate Demand Prediction

## 📘 Overview
This project aims to **predict real estate demand** across different sectors and months using **machine learning and time-series feature engineering**.  
By leveraging multiple real-world datasets (city indexes, property transactions, and search indices), the system builds a robust model that forecasts housing demand trends.

The entire process — from raw data ingestion to model training — is automated through a **modular end-to-end ML pipeline** designed for scalability and production-readiness.

---

## 🚀 Problem Statement
The real estate industry is heavily influenced by multiple factors such as:
- City-level economic indicators
- Search activity trends
- Land and housing transactions

Manually analyzing these variables is inefficient and error-prone.  
This project automates that process by:
- Collecting and merging large heterogeneous datasets
- Engineering temporal (rolling & lag) features
- Training predictive models using historical patterns

---

## 🎯 Objective
To build a **machine learning pipeline** that can:
1. Ingest multiple datasets and clean them automatically  
2. Engineer features like rolling means, lags, and cyclical encodings  
3. Scale numeric columns using `ColumnTransformer`  
4. Train multiple regression models using GridSearchCV  
5. Output the best-performing model with its evaluation metrics  

---

## 🧩 Architecture Overview


| Stage | Description |
|--------|--------------|
| **Data Ingestion** | Reads raw CSV files, merges multiple data sources, handles missing values, and creates a clean base dataset. |
| **Data Transformation** | Drops zero labels, applies scaling (StandardScaler / MinMaxScaler), encodes categorical columns if there any, and generates lag/rolling features. |
| **Model Training** | Trains and evaluates multiple models (CatBoost, RandomForest, XGBoost, etc.) using GridSearchCV. |
| **Evaluation** | Computes R², RMSE, and MAE metrics for model comparison. |
| **Artifacts** | Saves trained models and processed data into timestamped folders. |

---

## 🏗️ Folder Structure



---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Adigunolamilekannla/end_to_end_Real_Estate_Demand_Prediction.git
cd end_to_end_Real_Estate_Demand_Prediction

python3 -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)

pip install -r requirements.txt




