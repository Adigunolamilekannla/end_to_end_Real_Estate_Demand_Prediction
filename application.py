from flask import Flask, render_template, request
import pandas as pd
import  sys
from src.logging.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import load_object

app = Flask(__name__)

# Paths for model and preprocessor
MODEL_PATH = "final_model/model.pkl"
PROCESS_MODEL_PATH = "final_model/process_model.pkl"

# Load model and preprocessor
model = load_object(MODEL_PATH)
process_model = load_object(PROCESS_MODEL_PATH)

# Default values for 2000+ features (for file-based predictions)
default_features = {f"feature_{i}": 0 for i in range(1, 2001)}

# Top 10 important features for quick manual prediction
important_features = [
    "temperature", "pressure", "vibration", "voltage", "current",
    "humidity", "speed", "load", "torque", "power"
]

@app.route("/")
def home():
    """Render homepage with upload and quick prediction form"""
    return render_template("index.html", important_features=important_features)


@app.route("/predict", methods=["POST"])
def predict():
    """Handles prediction from uploaded CSV or manual input form"""
    try:
        # CASE 1: User uploaded a CSV file
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            data = pd.read_csv(file)
            logging.info(f"Uploaded CSV shape: {data.shape}")

            # ✅ Drop target column if it exists (important)
            if "nht_amount_new_house_transactions" in data.columns:
                data = data.drop(columns=["nht_amount_new_house_transactions"])

            # Transform with saved preprocessor
            transformed_data = process_model.transform(data)

            # Predict
            preds = model.predict(transformed_data)
            return render_template("result.html", predictions=preds.tolist(), mode="file")

        # CASE 2: Manual form entry
        else:
            input_data = {}
            for feature in important_features:
                value = request.form.get(feature, None)
                if value == "" or value is None:
                    input_data[feature] = 0.0
                else:
                    input_data[feature] = float(value)

            input_df = pd.DataFrame([input_data])

            # Transform input
            transformed_input = process_model.transform(input_df)

            pred = model.predict(transformed_input)[0]
            return render_template("result.html", predictions=[pred], mode="form")

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return render_template("result.html", error=str(e))



if __name__ == "__main__":
    app.run(host="0.0.0.0")
