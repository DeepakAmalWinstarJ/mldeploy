from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

import joblib

# Define the filename for the exported scaler
scaler_filename = 'scaler.joblib'

# Export the trained scaler object
joblib.dump(scaler, scaler_filename)

print(f"Trained StandardScaler successfully exported to {scaler_filename}")
# Load trained model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

# Feature order used during training
feature_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        # Convert JSON to dataframe
        input_df = pd.DataFrame([data])

        # Ensure column order matches training
        input_df = input_df[feature_columns]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        return jsonify({
            "prediction": int(prediction),
            "probability_no_diabetes": float(probability[0]),
            "probability_diabetes": float(probability[1])
        })

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) 
