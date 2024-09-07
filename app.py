from flask import Flask, request, jsonify
from predict import make_prediction
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = make_prediction(df)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)