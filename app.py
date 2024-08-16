import streamlit as st
import joblib
import numpy as np

# Global variables for model data
best_model = None
scaler = None
best_optimizer = None
accuracy = None

def load_model():
    global best_model, scaler, best_optimizer, accuracy
    try:
        model_data = joblib.load('model_data2.joblib')
        best_model = model_data['model']
        scaler = model_data['scaler']
        best_optimizer = model_data['best_optimizer']
        accuracy = model_data['accuracy']
    except FileNotFoundError:
        st.error("Error: The file 'model_data2.joblib' was not found.")
        raise
    except KeyError as e:
        st.error(f"Error: Missing key in the loaded model data: {e}")
        raise

def generate_form():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    features_list = []
    for i in range(st.session_state.num_rows):
        st.markdown(f'<div class="input-group"><h3>Row {i+1}</h3></div>', unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[0]:
            burden = st.number_input(f'Burden {i+1} (m)', min_value=0.0, key=f'burden_{i}')
            spacing = st.number_input(f'Spacing {i+1} (m)', min_value=0.0, key=f'spacing_{i}')
            ucs = st.number_input(f'UCS {i+1}', min_value=0.0, key=f'ucs_{i}')
        with cols[1]:
            hole_diameter = st.number_input(f'Hole Diameter {i+1} (mm)', min_value=0.0, key=f'hole_diameter_{i}')
            initial_stemming = st.number_input(f'Initial Stemming {i+1} (mm)', min_value=0.0, key=f'initial_stemming_{i}')
            final_stemming = st.number_input(f'Final Stemming {i+1} (mm)', min_value=0.0, key=f'final_stemming_{i}')
        with cols[2]:
            charge_length = st.number_input(f'Charge Length {i+1} (m)', min_value=0.0, key=f'charge_length_{i}')
            charge_per_hole = st.number_input(f'Charge per Hole {i+1}', min_value=0.0, key=f'charge_per_hole_{i}')
            powder_factor = st.number_input(f'Powder Factor {i+1}', min_value=0.0, key=f'powder_factor_{i}')

        features = [st.session_state[f'burden_{i}'], st.session_state[f'spacing_{i}'], st.session_state[f'ucs_{i}'], 
                    st.session_state[f'hole_diameter_{i}'], st.session_state[f'initial_stemming_{i}'], st.session_state[f'final_stemming_{i}'], 
                    st.session_state[f'charge_length_{i}'], st.session_state[f'charge_per_hole_{i}'], st.session_state[f'powder_factor_{i}']]
        features_list.append(features)

    if st.button('Predict'):
        try:
            final_features = scaler.transform(np.array(features_list))
            predictions = best_model.predict(final_features)

            predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            optimizer = best_optimizer['optimizer'] if isinstance(best_optimizer, dict) else str(best_optimizer)
            accuracy_value = float(accuracy) if isinstance(accuracy, (np.float32, np.float64)) else accuracy

            st.markdown('<div class="prediction-results">Prediction Results</div>', unsafe_allow_html=True)

            for i, pred in enumerate(predictions_list):
                st.write(f'Row {i+1} Prediction: {round(float(pred), 2)}')
            st.write(f'Best Optimizer: {optimizer}')
            st.write(f'Accuracy: {round(accuracy_value, 2)}')

        except Exception as e:
            st.error(f'Unexpected error: {str(e)}')

    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.markdown("""
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            }
            h1 {
                color: #004a99;
                text-align: center;
                font-family: 'Verdana', sans-serif;
                font-size: 36px;
                margin-bottom: 20px;
            }
            .input-group {
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }
            .input-group div {
                width: 30%;
                margin-bottom: 15px;
            }
            input[type="number"] {
                width: 100%;
                padding: 10px;
                margin: 5px 0;
                box-sizing: border-box;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            .stButton button {
                background-color: #004a99;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 18px;
                margin: 20px auto;
                cursor: pointer;
                border-radius: 10px;
                width: 100%;
            }
            .prediction-results {
                color: red;
                font-size: 24px;
                font-weight: bold;
                margin-top: 20px;
                text-align: center;
            }
            .form-container {
                margin-top: 40px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="container">', unsafe_allow_html=True)

    st.title("Rock Fragment Size Predictor")

    if 'num_rows' not in st.session_state:
        st.session_state.num_rows = 1

    if st.button('Generate Form'):
        load_model()
        generate_form()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
