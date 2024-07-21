import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

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
        
        print(f"Model type: {type(best_model)}")
        print(f"Scaler type: {type(scaler)}")
        print(f"Optimizer type: {type(best_optimizer)}")
        print(f"Accuracy type: {type(accuracy)}")
    except FileNotFoundError:
        app.logger.error("Error: The file 'model_data2.joblib' was not found.")
        raise
    except KeyError as e:
        app.logger.error(f"Error: Missing key in the loaded model data: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if best_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        num_rows = int(request.form['num_rows'])
        features_list = []
        
        for i in range(num_rows):
            features = [float(request.form[f'{feature}_{i}']) for feature in ['burden', 'spacing', 'ucs', 'hole_diameter', 'initial_stemming', 'final_stemming', 'charge_length', 'charge_per_hole', 'powder_factor']]
            features_list.append(features)
        
        final_features = scaler.transform(np.array(features_list))
        predictions = best_model.predict(final_features)
        
        # Convert any potential NumPy types to Python native types
        predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        optimizer = best_optimizer if isinstance(best_optimizer, str) else str(best_optimizer)
        accuracy_value = float(accuracy) if isinstance(accuracy, (np.float32, np.float64)) else accuracy
        
        return jsonify({
            'predictions': [round(float(pred), 2) for pred in predictions_list],
            'optimizer': optimizer,
            'accuracy': round(accuracy_value, 2)
        })
    except KeyError as e:
        return jsonify({'error': f'Missing input: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    load_model()  # Load the model when the app starts
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False') == 'True', host='0.0.0.0', port=5002)
