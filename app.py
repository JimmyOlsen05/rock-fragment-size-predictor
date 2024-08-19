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

def main():
    st.markdown("""
        <style>
            .main {
                padding: 0;
            }
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                color: #0000FF;
                text-align: center;
                font-size: 36px;
                margin-bottom: 20px;
            }
            .stButton>button {
                background-color: #0000FF;
                color: white;
                font-size: 16px;
                padding: 10px 24px;
                border-radius: 20px;
                border: none;
                cursor: pointer;
            }
            .row-box {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .row-box h3 {
                margin-top: 0;
            }
            .stNumberInput>div>div>input {
                text-align: left;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Rock Fragment Size Predictor")

    load_model()

    if best_model is None:
        st.error('Model not loaded')
        return

    if 'num_rows' not in st.session_state:
        st.session_state.num_rows = 5
    if 'form_generated' not in st.session_state:
        st.session_state.form_generated = False

    num_rows = st.number_input('Number of rows to predict', min_value=1, max_value=1000, value=st.session_state.num_rows, key='num_rows_input')

    if st.button('Generate Form') or st.session_state.form_generated:
        st.session_state.form_generated = True
        st.session_state.num_rows = num_rows
        features_list = []

        for i in range(0, num_rows, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < num_rows:
                    with cols[j]:
                        st.markdown(f'<div class="row-box">', unsafe_allow_html=True)
                        st.subheader(f'Row {i+j+1}')
                        burden = st.number_input(f'Burden (m):', key=f'burden_{i+j}')
                        spacing = st.number_input(f'Spacing (m):', key=f'spacing_{i+j}')
                        ucs = st.number_input(f'UCS:', key=f'ucs_{i+j}')
                        hole_diameter = st.number_input(f'Hole Diameter (mm):', key=f'hole_diameter_{i+j}')
                        initial_stemming = st.number_input(f'Initial Stemming Height (m):', key=f'initial_stemming_{i+j}')
                        final_stemming = st.number_input(f'Final Stemming Height (m):', key=f'final_stemming_{i+j}')
                        charge_length = st.number_input(f'Charge Length Height (m):', key=f'charge_length_{i+j}')
                        charge_per_hole = st.number_input(f'Charge per Hole (Kg):', key=f'charge_per_hole_{i+j}')
                        powder_factor = st.number_input(f'Powder Factor (Kg/m³):', key=f'powder_factor_{i+j}')

                        features = [burden, spacing, ucs, hole_diameter, initial_stemming, final_stemming, charge_length, charge_per_hole, powder_factor]
                        features_list.append(features)
                        st.markdown('</div>', unsafe_allow_html=True)

        if st.button('Predict'):
            try:
                final_features = scaler.transform(np.array(features_list))
                predictions = best_model.predict(final_features)

                predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
                optimizer = best_optimizer['optimizer'] if isinstance(best_optimizer, dict) else str(best_optimizer)
                accuracy_value = float(accuracy) if isinstance(accuracy, (np.float32, np.float64)) else accuracy

                st.subheader('Prediction Results')

                for i, pred in enumerate(predictions_list):
                    st.write(f'Row {i+1} Prediction: {round(float(pred), 2)}')
                st.write(f'Best Optimizer: {optimizer}')
                st.write(f'Accuracy: {round(accuracy_value, 2)}')

            except Exception as e:
                st.error(f'Unexpected error: {str(e)}')

if __name__ == '__main__':
    main()
