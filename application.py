from flask import Flask, render_template, request
import pandas as pd
import sys
from src.logging.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import load_object

# ==================================================
# Flask App
# ==================================================
app = Flask(__name__)

# ==================================================
# Paths for model and preprocessor
# ==================================================
MODEL_PATH = "final_model/model.pkl"
PROCESS_MODEL_PATH = "final_model/process_model.pkl"

# ==================================================
# Load model and preprocessor
# ==================================================
try:
    model = load_object(MODEL_PATH)
    process_model = load_object(PROCESS_MODEL_PATH)
    logging.info("✅ Model and preprocessor loaded successfully.")
except Exception as e:
    raise CustomException(e, sys)

# ==================================================
# Routes
# ==================================================
@app.route("/")
def home():
    """Render homepage with upload and quick prediction form"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handles prediction from uploaded CSV or manual input form"""
    try:
        # ✅ CASE 1: CSV File Upload
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            data = pd.read_csv(file)
            logging.info(f"Uploaded CSV shape: {data.shape}")

            # Drop target column if present
            if "nht_amount_new_house_transactions" in data.columns:
                data = data.drop(columns=["nht_amount_new_house_transactions"])

            transformed_data = process_model.transform(data)
            preds = model.predict(transformed_data)

            return render_template(
                "result.html",
                predictions=preds.tolist(),
                mode="file"
            )

       
    except Exception as e:
        logging.error(f"❌ Prediction failed: {str(e)}")
        return render_template("result.html", error=str(e))


# ==================================================
# Run app (for local and Elastic Beanstalk)
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
