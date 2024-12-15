from flask import render_template, request, jsonify
from stronka import db, app
from stronka.model_LSTM import scaler, model
import torch
import numpy as np
import logging
from sqlalchemy import text

class Dataset(db.Model):
    __tablename__ = 'datasets'
    Date = db.Column('Date', db.Text, primary_key=True)
    CO = db.Column('CO (ppm)', db.Float, nullable=False)
    NO2 = db.Column('NO2 (ppb)', db.Float, nullable=False)
    O3 = db.Column('O3 (ppm)', db.Float, nullable=False)
    PM25 = db.Column('PM2.5 (ug/m3 LC)', db.Float, nullable=False)
    SO2 = db.Column('SO2 (ppb)', db.Float, nullable=False)
    City = db.Column('City', db.Text, nullable=False)

class Predictions(db.Model):
    __tablename__ = 'predictions'
    Date = db.Column('Date', db.Text, primary_key=True)
    CO = db.Column('CO (ppm)', db.Float, nullable=False)
    NO2 = db.Column('NO2 (ppb)', db.Float, nullable=False)
    O3 = db.Column('O3 (ppm)', db.Float, nullable=False)
    PM25 = db.Column('PM2.5 (ug/m3 LC)', db.Float, nullable=False)
    SO2 = db.Column('SO2 (ppb)', db.Float, nullable=False)
    City = db.Column('City', db.Text, nullable=False)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

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

    o3_values = np.array(o3_values).reshape(-1, 1)

    o3_values_scaled = scaler.transform(o3_values)

    o3_values_scaled = o3_values_scaled.reshape(1, sequence_length, 1)

    input_tensor = torch.tensor(o3_values_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    prediction_scaled = prediction_scaled.numpy()
    prediction = scaler.inverse_transform(prediction_scaled)

    prediction_list = prediction.flatten().tolist()

    response = {
        'predictions': prediction_list
    }
    return jsonify(response)

@app.route('/api/data/live', methods=['GET'])
def get_live_data():
    city = request.args.get('city', default='', type=str)
    start = request.args.get('start', default='', type=str)
    metric = request.args.get('metric', default='', type=str)
    end = request.args.get('end', default='', type=str)

    datasets_sql = f"""
        SELECT "Date", "{metric}"
        FROM datasets
        WHERE "City" = :city
        AND "Date" >= :start
        AND "Date" <= :end
        ORDER BY "Date" DESC
    """
    datasets_results = db.session.execute(text(datasets_sql), {'city': city, 'start': start, 'end': end}).fetchall()

    predictions_sql = f"""
        SELECT "Date", "{metric}" AS "predicted_{metric}"
        FROM predictions
        WHERE "City" = :city
        AND "Date" >= :start
        AND "Date" <= :end
        ORDER BY "Date" DESC
    """
    predictions_results = db.session.execute(text(predictions_sql), { 'city': city, 'start': start, 'end': end}).fetchall()

    datasets_dict = {row['Date']: {"date": row['Date'], metric: row[metric]} for row in datasets_results}
    predictions_dict = {row['Date']: {f"predicted_{metric}": row[f"predicted_{metric}"]} for row in predictions_results}

    combined_results = []
    for date, dataset_row in datasets_dict.items():
        combined_row = dataset_row
        if date in predictions_dict:
            combined_row.update(predictions_dict[date])
        combined_results.append(combined_row)

    return jsonify(combined_results)
