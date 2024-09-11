from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model and Scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the features used by the model
features = ['Year', 'Month', 'Day', 'Min_Temp', 'Rainfall', 'Humidity', 'Wind_Direction', 'Wind_Speed', 'timestamp', 'Latitude', 'Longitude', 'Cluster']

# Prediction route for bulk data
@app.route('/predict-bulk', methods=['POST'])
def predict_bulk():
    try:
        # Get data from the POST request (it should be a list of records)
        data = request.get_json()

        # Validate that data is a list of dictionaries
        if not isinstance(data, list):
            return jsonify({'error': 'Input data should be a list of dictionaries'}), 400

        # Prepare input data for prediction
        input_df = pd.DataFrame(data)

        # Ensure all required features are present (excluding 'timestamp' which will be added)
        if not all(col in input_df.columns for col in features if col != 'timestamp'):
            return jsonify({'error': 'Missing required features in some data entries'}), 400

        # Calculate 'timestamp' for each row
        input_df['timestamp'] = input_df.apply(lambda row: pd.Timestamp(row['Year'], row['Month'], row['Day']).timestamp(), axis=1)

        # Scale the input features
        X_scaled = scaler.transform(input_df[features])

        # Predict using the Random Forest model
        predictions = model.predict(X_scaled)

        # Return the predictions for all input rows
        input_df['predicted_temperature'] = predictions
        result = input_df[['Year', 'Month', 'Day', 'Latitude', 'Longitude', 'predicted_temperature']].to_dict(orient='records')

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
