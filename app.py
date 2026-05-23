import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open('boston_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaling.pkl', 'rb') as f:
    scalar = pickle.load(f)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    print(data)
    features = data['data']
    data_scaled = scalar.transform(
        np.array([list(feature.values()) for feature in features])
        )
    print(data_scaled)
    output = model.predict(data_scaled)
    print(output)
    return jsonify({'prediction': output[0]})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text = f"The House price prediction is: {output}".format(output))

if __name__ == '__main__':
    app.run(debug=True)