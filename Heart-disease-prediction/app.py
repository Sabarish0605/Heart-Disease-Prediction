from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging
from sklearn.exceptions import NotFittedError

app = Flask(__name__)

# Load the trained model
try:
    with open('ensemble_model.pkl', 'rb') as model_file:
        ensemble_model = pickle.load(model_file)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        logging.info(f"Received data: {data}")
        
        # Validate input data
        required_fields = ['age', 'cholesterol', 'bp', 'sex_Female', 'sex_Male', 
                         'diabetes_No', 'diabetes_Yes', 'age_cholesterol', 'bp_cholesterol']
        
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Get both class prediction and probabilities
        prediction = ensemble_model.predict(input_data)[0]
        probabilities = ensemble_model.predict_proba(input_data)[0].tolist()
        
        logging.info(f"Prediction: {prediction}, Probabilities: {probabilities}")
        
        return jsonify({
            'prediction': int(prediction),
            'probabilities': probabilities,
            'confidence': max(probabilities)
        })
        
    except NotFittedError:
        return jsonify({"error": "Model not properly trained"}), 500
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)