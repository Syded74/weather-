@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request (it should be an array of input data)
        data = request.get_json()

        # Validate that data is a list of dictionaries
        if not isinstance(data, list):
            return jsonify({'error': 'Input data should be a list of dictionaries'}), 400

        # Prepare input data for prediction
        input_df = pd.DataFrame(data)

        # Ensure all required features are present
        if not all(col in input_df.columns for col in features):
            return jsonify({'error': 'Missing required features in some data entries'}), 400

        # Scale the input features
        X_scaled = scaler.transform(input_df[features])

        # Predict using the Random Forest model
        predictions = model.predict(X_scaled)

        # Return the predictions for all input rows
        return jsonify({
            'predictions': predictions.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
