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
    st.title("Rock Fragment Size Predictor")

    load_model()

    if best_model is None:
        st.error('Model not loaded')
        return

    num_rows = st.number_input('Number of rows to predict', min_value=1, value=1)

    features_list = []
    for i in range(num_rows):
        st.write(f'Row {i+1}')
        burden = st.number_input(f'Burden {i+1}', min_value=0.0)
        spacing = st.number_input(f'Spacing {i+1}', min_value=0.0)
        ucs = st.number_input(f'UCS {i+1}', min_value=0.0)
        hole_diameter = st.number_input(f'Hole Diameter {i+1}', min_value=0.0)
        initial_stemming = st.number_input(f'Initial Stemming {i+1}', min_value=0.0)
        final_stemming = st.number_input(f'Final Stemming {i+1}', min_value=0.0)
        charge_length = st.number_input(f'Charge Length {i+1}', min_value=0.0)
        charge_per_hole = st.number_input(f'Charge per Hole {i+1}', min_value=0.0)
        powder_factor = st.number_input(f'Powder Factor {i+1}', min_value=0.0)

        features = [burden, spacing, ucs, hole_diameter, initial_stemming, final_stemming, charge_length, charge_per_hole, powder_factor]
        features_list.append(features)

    if st.button('Predict'):
        try:
            final_features = scaler.transform(np.array(features_list))
            predictions = best_model.predict(final_features)

            # Extract the optimizer name
            optimizer_name = best_optimizer['optimizer'] if isinstance(best_optimizer, dict) and 'optimizer' in best_optimizer else str(best_optimizer)
            accuracy_value = float(accuracy) if isinstance(accuracy, (np.float32, np.float64)) else accuracy

            st.write('Prediction Results')
            for i, pred in enumerate(predictions):
                st.write(f'Row {i+1} Prediction: {round(float(pred), 2)}')
            st.write(f'Best Optimizer: {optimizer_name}')
            st.write(f'Accuracy: {round(accuracy_value, 2)}')
        except Exception as e:
            st.error(f'Unexpected error: {str(e)}')

if __name__ == '__main__':
    main()
