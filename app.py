from flask import Flask, request, jsonify
from predict import make_prediction
import pandas as pd
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = make_prediction(df)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    # DigitalOcean App Platform uses PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)