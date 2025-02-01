import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar modelo entrenado
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        data = np.array(data).reshape(1, -1)  # Convertir a la forma adecuada
        prediction = model.predict(data)
        return jsonify({'prediction': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
