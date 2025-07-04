from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

feature_columns = [
    'month', 'origin', 'departure',
    'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4',
    'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7'
]

model_path = 'decisionTree.pkl'
model = None

def generate_varied_data():
    np.random.seed(42)
    data = []
    labels = []

    for _ in range(200):
        month = np.random.randint(1, 13)
        origin = np.random.randint(1, 6)
        departure = np.random.randint(0, 2400)
        quarter = (month - 1) // 3 + 1
        day = np.random.randint(1, 8)

        input_dict = {
            'month': month,
            'origin': origin,
            'departure': departure,
            **{f'quarter_{i}': int(i == quarter) for i in range(1, 5)},
            **{f'day_{i}': int(i == day) for i in range(1, 8)},
        }

        # Add noise to make the label less predictable
        label = int((month + day + (departure % 1000) + np.random.randint(0, 10)) % 2)

        data.append(input_dict)
        labels.append(label)

    X_train = pd.DataFrame(data)
    y_train = np.array(labels)

    return X_train, y_train

def train_and_save_model():
    global model
    X, y = generate_varied_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    logging.info("Model trained and saved.")

def load_model():
    global model
    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded model type: {type(model)}")
        # If the loaded object is not a RandomForestClassifier, retrain and save
        if not isinstance(model, RandomForestClassifier):
            logging.warning("Loaded object is not a RandomForestClassifier. Retraining model.")
            train_and_save_model()
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}. Retraining model.")
        train_and_save_model()

# Initialize model
if not os.path.exists(model_path):
    train_and_save_model()
else:
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        "model_ready": model is not None,
        "model_type": str(type(model))
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not available"}), 500

    try:
        # Log received form data for debugging
        logging.info(f"Received form data: {dict(request.form)}")
        
        # Input validation with better error handling
        try:
            month = int(request.form.get('month', 0))
            day = int(request.form.get('day', 0))
            quarter = int(request.form.get('quarter', 0))
            origin = int(request.form.get('origin', 0))
            departure = int(request.form.get('departure', 0))
        except (ValueError, TypeError) as e:
            logging.error(f"Type conversion error: {str(e)}")
            return jsonify({"success": False, "error": "All fields must be valid numbers"}), 400
        
        # Log parsed values for debugging
        logging.info(f"Parsed values - month: {month}, day: {day}, quarter: {quarter}, origin: {origin}, departure: {departure}")
        
        # Validation with detailed error messages
        if not (1 <= month <= 12):
            return jsonify({"success": False, "error": "Month must be between 1 and 12"}), 400
        if not (1 <= day <= 7):
            return jsonify({"success": False, "error": "Day must be between 1 and 7"}), 400
        if not (1 <= quarter <= 4):
            return jsonify({"success": False, "error": "Quarter must be between 1 and 4"}), 400
        if not (1 <= origin <= 5):
            return jsonify({"success": False, "error": f"Origin must be between 1 and 5, received: {origin}"}), 400
        if not (0 <= departure <= 2359):
            return jsonify({"success": False, "error": "Departure time must be between 0 and 2359 (HHMM format)"}), 400

        # Create input dictionary
        input_dict = {
            'month': month,
            'origin': origin,
            'departure': departure,
            **{f'quarter_{i}': int(i == quarter) for i in range(1, 5)},
            **{f'day_{i}': int(i == day) for i in range(1, 8)},
        }

        # Create DataFrame with correct column order
        X_input = pd.DataFrame([input_dict], columns=feature_columns)
        
        # Make prediction
        prediction = int(model.predict(X_input)[0])
        proba = model.predict_proba(X_input)[0]
        confidence = float(proba[prediction])

        logging.info(f"Prediction successful - prediction: {prediction}, confidence: {confidence}")

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify the prediction function works"""
    try:
        # Test with sample data
        test_data = {
            'month': 6,
            'day': 3,
            'quarter': 2,
            'origin': 2,
            'departure': 1200
        }
        
        input_dict = {
            'month': test_data['month'],
            'origin': test_data['origin'],
            'departure': test_data['departure'],
            **{f'quarter_{i}': int(i == test_data['quarter']) for i in range(1, 5)},
            **{f'day_{i}': int(i == test_data['day']) for i in range(1, 8)},
        }

        X_input = pd.DataFrame([input_dict], columns=feature_columns)
        prediction = int(model.predict(X_input)[0])
        proba = model.predict_proba(X_input)[0]
        confidence = float(proba[prediction])

        return jsonify({
            "success": True,
            "test_data": test_data,
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "message": "Test successful"
        })

    except Exception as e:
        logging.error(f"Test error: {str(e)}")
        return jsonify({"success": False, "error": f"Test failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)