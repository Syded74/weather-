from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the models and the scaler with error handling
try:
    random_forest_model = joblib.load('random_forest_model.joblib')
    gradient_boosting_model = joblib.load('gradient_boosting_model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    print(f"Error loading models or scaler: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()

        # Preprocess the input
        features = data['features']
        scaled_features = scaler.transform([features])

        # Make predictions using both models
        rf_prediction = random_forest_model.predict(scaled_features)
        gb_prediction = gradient_boosting_model.predict(scaled_features)

        return jsonify({
            'random_forest_prediction': rf_prediction.tolist(),
            'gradient_boosting_prediction': gb_prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
