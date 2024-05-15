from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

model = joblib.load('iris_classifier.joblib')
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]).reshape(1, -1)
    scaled_features = scaler.fit_transform(features)
    prediction = model.predict(scaled_features)

    iris_species = ['setosa', 'versicolor', 'virginica']
    response = {'species': iris_species[prediction[0]]}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
