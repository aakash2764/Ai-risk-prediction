from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import tempfile
import os
from predict import HealthcarePredictor

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = HealthcarePredictor()
predictor.load_trained_models()

@app.route('/predict', methods=['POST'])
def predict_patient():
    try:
        # Get patient data from frontend
        patient_data = request.json
        
        # Convert to DataFrame format
        df_data = []
        for entry in patient_data['data']:
            row = {'PatientID': patient_data['PatientID']}
            row.update(entry)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        # Make predictions
        predictions = predictor.make_predictions(temp_file)
        
        # Clean up temp file
        os.unlink(temp_file)
        
        # Return prediction results
        return jsonify(predictions.to_dict('records')[0])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Handle CSV file upload
        file = request.files['file']
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
            file.save(f.name)
            temp_file = f.name
        
        # Make predictions
        predictions = predictor.make_predictions(temp_file)
        
        # Save results
        output_file, summary = predictor.save_predictions(predictions)
        
        # Clean up
        os.unlink(temp_file)
        
        return jsonify({
            'predictions': predictions.to_dict('records'),
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)