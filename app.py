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

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f5;
        color: #444;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stNumberInput>div>input {
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    .stTextInput>div>input {
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    .prediction-result {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    .title {
        color: #4CAF50;
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #333;
        font-size: 24px;
        margin-top: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    .input-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 10px;
    }
    .input-row > div {
        flex: 1;
        min-width: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.markdown("<div class='title'>Rock Fragment Size Predictor</div>", unsafe_allow_html=True)

    load_model()

    if best_model is None:
        st.error('Model not loaded')
        return

    num_rows = st.number_input('Number of rows to predict', min_value=1, value=1)

    features_list = []
    for i in range(num_rows):
        st.markdown(f"<div class='subtitle'>Row {i+1}</div>", unsafe_allow_html=True)
        with st.container():
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
            col7, col8, col9 = st.columns(3)

            with col1:
                burden = st.number_input(f'Burden {i+1}', min_value=0.0)
            with col2:
                spacing = st.number_input(f'Spacing {i+1}', min_value=0.0)
            with col3:
                ucs = st.number_input(f'UCS {i+1}', min_value=0.0)
            with col4:
                hole_diameter = st.number_input(f'Hole Diameter {i+1}', min_value=0.0)
            with col5:
                initial_stemming = st.number_input(f'Initial Stemming {i+1}', min_value=0.0)
            with col6:
                final_stemming = st.number_input(f'Final Stemming {i+1}', min_value=0.0)
            with col7:
                charge_length = st.number_input(f'Charge Length {i+1}', min_value=0.0)
            with col8:
                charge_per_hole = st.number_input(f'Charge per Hole {i+1}', min_value=0.0)
            with col9:
                powder_factor = st.number_input(f'Powder Factor {i+1}', min_value=0.0)

        features = [burden, spacing, ucs, hole_diameter, initial_stemming, final_stemming, charge_length, charge_per_hole, powder_factor]
        features_list.append(features)

    if st.button('Predict'):
        try:
            final_features = scaler.transform(np.array(features_list))
            predictions = best_model.predict(final_features)

            predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            optimizer_name = best_optimizer['optimizer'] if isinstance(best_optimizer, dict) and 'optimizer' in best_optimizer else str(best_optimizer)
            accuracy_value = float(accuracy) if isinstance(accuracy, (np.float32, np.float64)) else accuracy

            st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
            st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
            for i, pred in enumerate(predictions_list):
                st.write(f'Row {i+1} Prediction: {round(float(pred), 2)}')
            st.write(f'Best Optimizer: {optimizer_name}')
            st.write(f'Accuracy: {round(accuracy_value, 2)}')
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f'Unexpected error: {str(e)}')

if __name__ == '__main__':
    main()
