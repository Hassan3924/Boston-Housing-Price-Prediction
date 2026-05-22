import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open('boston_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaling.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    features = data['data']
    data_scaled = scaler.transform(
        np.array([list(feature.values()) for feature in features])
        )
    print(data_scaled)
    output = model.predict(data_scaled)
    print(output)
    return jsonify({'prediction': output[0]})

if __name__ == '__main__':
    app.run(debug=True)