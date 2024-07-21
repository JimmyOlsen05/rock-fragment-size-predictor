import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model_data = joblib.load('model_data2.joblib')
best_model = model_data['model']
scaler = model_data['scaler']
best_optimizer = model_data['best_optimizer']
accuracy = model_data['accuracy']

# Custom CSS
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }

        .container {
            width: 80%;
            margin: auto;
            overflow: hidden;
            padding: 50px;
            background-color: #fff;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            margin-top: 20px;
        }

        h1 {
            color: darkblue;
            text-align: center;
            font-family: cursive;
        }

        .input-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            color: #666;
        }

        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        button {
            display: block;
            width: 220px;
            margin: 20px auto;
            padding: 15px;
            background-color: blue;
            color: #fff;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 10px 10px 10px grey;
        }

        button:hover {
            background-color: darkred;
        }

        #result {
            margin-top: 30px;
            padding: 20px;
            background-color: #e9f7ef;
            border-radius: 4px;
        }

        #result h2 {
            color: #2c3e50;
            margin-bottom: 15px;
        }

        #result p {
            margin-bottom: 10px;
            color: #34495e;
        }

        .hidden {
            display: none;
        }

        .rows-container {
            display: flex;
            flex-wrap: nowrap;
            overflow-x: auto;
            gap: 20px;
            padding-bottom: 20px;
        }

        .row {
            flex: 0 0 auto;
            width: 300px;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }

        .row h3 {
            margin-top: 0;
            color: #333;
            text-align: center;
        }

        .input-group {
            margin-bottom: 10px;
        }

        .input-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }

        .input-group input {
            width: 100%;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }

        #form-container {
            margin-top: 20px;
        }

        #num_rows {
            width: 60px;
            margin-right: 10px;
        }

        #predictions-container {
            margin-bottom: 20px;
        }

        #predictions-container p {
            margin: 5px 0;
        }

        button[type="submit"] {
            display: block;
            margin: 20px auto 0;
        }
    </style>
""", unsafe_allow_html=True)

# App layout
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<h1>Rock Fragment Size Predictor</h1>', unsafe_allow_html=True)

num_rows = st.number_input('Number of rows to predict:', min_value=1, value=1)
if st.button('Generate Form'):
    features_list = []
    form_container = st.container()

    with form_container:
        for i in range(num_rows):
            st.markdown(f'<div class="row"><h3>Row {i + 1}</h3>', unsafe_allow_html=True)
            burden = st.number_input(f'Burden (m) for Row {i + 1}:', step=0.1, format="%.1f", key=f'burden_{i}')
            spacing = st.number_input(f'Spacing (m) for Row {i + 1}:', step=0.1, format="%.1f", key=f'spacing_{i}')
            ucs = st.number_input(f'UCS for Row {i + 1}:', step=0.1, format="%.1f", key=f'ucs_{i}')
            hole_diameter = st.number_input(f'Hole Diameter (mm) for Row {i + 1}:', step=0.1, format="%.1f", key=f'hole_diameter_{i}')
            initial_stemming = st.number_input(f'Initial Stemming Height (m) for Row {i + 1}:', step=0.1, format="%.1f", key=f'initial_stemming_{i}')
            final_stemming = st.number_input(f'Final Stemming Height (m) for Row {i + 1}:', step=0.1, format="%.1f", key=f'final_stemming_{i}')
            charge_length = st.number_input(f'Charge Length (m) for Row {i + 1}:', step=0.1, format="%.1f", key=f'charge_length_{i}')
            charge_per_hole = st.number_input(f'Charge / Hole (Kg) for Row {i + 1}:', step=0.1, format="%.1f", key=f'charge_per_hole_{i}')
            powder_factor = st.number_input(f'Powder Factor (Kg/m3) for Row {i + 1}:', step=0.01, format="%.2f", key=f'powder_factor_{i}')
            st.markdown('</div>', unsafe_allow_html=True)
            features = [burden, spacing, ucs, hole_diameter, initial_stemming, final_stemming, charge_length, charge_per_hole, powder_factor]
            features_list.append(features)

    if st.button('Predict'):
        final_features = scaler.transform(np.array(features_list))
        predictions = best_model.predict(final_features)

        st.markdown('<div id="result">', unsafe_allow_html=True)
        st.markdown('<h2>Prediction Results</h2>', unsafe_allow_html=True)
        for i, pred in enumerate(predictions):
            st.markdown(f'<p>Row {i + 1} Predicted Fragment Size: {round(pred, 2)}</p>', unsafe_allow_html=True)
        st.markdown(f'<p>Best Optimizer: {best_optimizer}</p>', unsafe_allow_html=True)
        st.markdown(f'<p>Accuracy: {round(accuracy, 2)}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
