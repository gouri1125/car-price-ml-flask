from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import sklearn

app = Flask(__name__)

# Load trained model
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'car_price_model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])
        owner = int(request.form['owner'])

        fuel_type = request.form['fuel']
        seller_type = request.form['seller']
        transmission = request.form['transmission']

        # Encoding
        fuel_diesel = 1 if fuel_type == 'Diesel' else 0
        fuel_petrol = 1 if fuel_type == 'Petrol' else 0
        seller_individual = 1 if seller_type == 'Individual' else 0
        transmission_manual = 1 if transmission == 'Manual' else 0

        # Create input dataframe
        input_data = pd.DataFrame({
            'Year': [year],
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Owner': [owner],
            'Fuel_Type_Diesel': [fuel_diesel],
            'Fuel_Type_Petrol': [fuel_petrol],
            'Seller_Type_Individual': [seller_individual],
            'Transmission_Manual': [transmission_manual]
        })

        # Align columns with model
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_data)[0]


    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

