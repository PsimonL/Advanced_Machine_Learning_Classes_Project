from flask import render_template, request, jsonify
from stronka import app
from stronka.model_LSTM import scaler, model
import torch
import numpy as np

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('graphs.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sequence_length = 3

    if not data or 'o3_values' not in data:
        return jsonify({'error': 'No data provided or wrong key. Expecting JSON with key "co_values".'}), 400

    o3_values = data['o3_values']

    if len(o3_values) != sequence_length:
        return jsonify({'error': f'Expected {sequence_length} O3 values.'}), 400

    # Convert to numpy array and reshape
    o3_values = np.array(o3_values).reshape(-1, 1)

    # Scale the input data
    o3_values_scaled = scaler.transform(o3_values)

    # Reshape for the model input
    o3_values_scaled = o3_values_scaled.reshape(1, sequence_length, 1)

    # Convert to tensor
    input_tensor = torch.tensor(o3_values_scaled, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    # Inverse scale the prediction
    prediction_scaled = prediction_scaled.numpy()
    prediction = scaler.inverse_transform(prediction_scaled)

    # Convert prediction to list
    prediction_list = prediction.flatten().tolist()

    # Prepare the response
    response = {
        'predictions': prediction_list
    }
    return jsonify(response)
