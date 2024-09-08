import logging
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def hello():
    app.logger.info("Hello route called")
    return "Hello, World!"

# Load models and scaler
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    random_forest_model = joblib.load(os.path.join(current_dir, 'random_forest_model.joblib'))
    gradient_boosting_model = joblib.load(os.path.join(current_dir, 'gradient_boosting_model.joblib'))
    scaler = joblib.load(os.path.join(current_dir, 'scaler.joblib'))
    app.logger.info("Models loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading models: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Predict route called")
    # Get the data from the request
    data = request.get_json()
    app.logger.debug(f"Received data: {data}")

    # Extract the features from the data
    try:
        features = []
        for entry in data['features']:
            feature_row = [
                float(entry['Year']),
                float(entry['Month']),
                float(entry['Day']),
                float(entry['Min_Temp']),
                float(entry['Rainfall']),
                float(entry['Humidity']),
                float(entry['Wind_Direction']),
                float(entry['Wind_Speed']),
                float(entry['timestamp']),  # Ensure timestamp is included
                float(entry['Latitude']),
                float(entry['Longitude']),
                float(entry['Cluster'])
            ]
            features.append(feature_row)

        # Convert the list of features into a numpy array
        features = np.array(features)
        app.logger.debug(f"Processed features shape: {features.shape}")

        # Scale the features
        scaled_features = scaler.transform(features)
        app.logger.debug(f"Scaled features shape: {scaled_features.shape}")

        # Make predictions using both models
        rf_prediction = random_forest_model.predict(scaled_features)
        gb_prediction = gradient_boosting_model.predict(scaled_features)

        # Combine the predictions
        final_predictions = rf_prediction + gb_prediction
        app.logger.info(f"Predictions made successfully. Shape: {final_predictions.shape}")

        # Return the predictions as a JSON response
        return jsonify({'Predicted_Max_Temp': final_predictions.tolist()})

    except KeyError as ke:
        error_msg = f"Missing key in input data: {str(ke)}"
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except ValueError as ve:
        error_msg = f"Invalid value in input data: {str(ve)}"
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')