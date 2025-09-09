import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
from typing import Dict, List, Tuple, Any
import json
import shap
from datetime import datetime
import os


warnings.filterwarnings('ignore')

class HealthcarePredictor:
    def __init__(self):
        """Initialize predictor with trained models and feature engineering logic"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))  # points to backend/
        self.models_path = os.path.join(self.base_path, 'models')
        self.reports_dir = os.path.join(self.base_path, 'high_risk_patient_reports')

        self.deterioration_model = None
        self.riskiness_model = None
        self.explainers = None
        self.feature_importance = None
        self.clinical_explanations = None
        
        # Clinical bands for feature engineering (copied from dataset.py)
        self.clinical_bands = {
            'SystolicBP': {'healthy': (110, 125), 'low': (126, 140), 'medium': (141, 160), 'severe': (161, 200)},
            'DiastolicBP': {'healthy': (65, 80), 'low': (81, 90), 'medium': (91, 100), 'severe': (101, 120)},
            'HeartRate': {'healthy': (65, 85), 'low': (86, 100), 'medium': (101, 120), 'severe': (121, 160)},
            'RespirationRate': {'healthy': (12, 18), 'low': (19, 20), 'medium': (21, 24), 'severe': (25, 40)},
            'O2Sat': {'healthy': (96, 99), 'low': (94, 95), 'medium': (91, 93), 'severe': (75, 90)},
            'Glucose': {'healthy': (80, 110), 'low': (111, 139), 'medium': (140, 199), 'severe': (200, 500)},
            'MedicationAdherence': {'healthy': (95, 100), 'low': (90, 94), 'medium': (70, 89), 'severe': (0, 69)},
            'WeightBMI': {'healthy': (20, 26), 'low': (27, 29), 'medium': (30, 34), 'severe': (35, 60)},
            'ALT': {'healthy': (10, 40), 'low': (41, 60), 'medium': (61, 80), 'severe': (81, 200)},
            'AST': {'healthy': (10, 40), 'low': (41, 60), 'medium': (61, 80), 'severe': (81, 200)},
            'Bilirubin': {'healthy': (0.3, 1.0), 'low': (1.1, 1.2), 'medium': (1.3, 2.0), 'severe': (2.1, 30)},
            'Albumin': {'healthy': (3.8, 5.0), 'low': (3.5, 3.7), 'medium': (3.0, 3.4), 'severe': (2.0, 2.9)},
            'Creatinine': {'healthy': (0.7, 1.1), 'low': (1.2, 1.4), 'medium': (1.5, 2.0), 'severe': (2.1, 12)},
            'Sodium': {'healthy': (136, 142), 'low': (133, 135), 'medium': (130, 132), 'severe': (120, 129)},
            'Potassium': {'healthy': (3.8, 4.8), 'low': (3.5, 3.7), 'medium': (3.0, 3.4), 'severe': (2.5, 2.9)},
            'Cholesterol': {'healthy': (160, 190), 'low': (191, 199), 'medium': (200, 239), 'severe': (240, 400)},
            'SleepHours': {'healthy': (7, 9), 'low': (6.5, 7), 'medium': (6, 6.4), 'severe': (4, 5.9)},
            'ExerciseMinutes': {'healthy': (25, 45), 'low': (17, 24), 'medium': (8, 16), 'severe': (0, 7)},
            'AlcoholIntake': {'healthy': (0, 2), 'low': (3, 5), 'medium': (6, 10), 'severe': (11, 20)},
            'SmokingCigs': {'healthy': (0, 0), 'low': (1, 5), 'medium': (6, 10), 'severe': (11, 60)}
        }
        
        # Feature engineering configuration (from preprocess.py)
        self.feature_config = {
            'vitals': {
                'features': ['SystolicBP', 'DiastolicBP', 'HeartRate', 'RespirationRate', 'O2Sat', 'WeightBMI'],
                'windows': [7, 14, 30, 60, 90, 180],
                'aggregations': ['mean', 'min', 'max', 'std', 'last', 'slope', 'ema']
            },
            'labs': {
                'features': ['Glucose', 'Cholesterol', 'Creatinine', 'Sodium', 'Potassium', 'Albumin', 'AST', 'ALT', 'Bilirubin'],
                'windows': [30, 90, 180],
                'aggregations': ['mean', 'min', 'max', 'std', 'last', 'ema']
            },
            'medication': {
                'features': ['MedicationAdherence'],
                'windows': [7, 14, 30, 60, 180],
                'aggregations': ['mean', 'min', 'max', 'last', 'slope', 'adherence_below_80_pct']
            },
            'lifestyle': {
                'features': ['SleepHours', 'ExerciseMinutes', 'AlcoholIntake', 'SmokingCigs'],
                'windows': [30, 60, 180],
                'aggregations': ['mean', 'max', 'slope', 'ema']
            }
        }

    def load_trained_models(self):
        """Load the trained models and explanations"""
        print("Loading trained models...")
        
        try:
            self.deterioration_model = joblib.load(os.path.join(self.models_path, 'deterioration_model.pkl'))
            self.riskiness_model = joblib.load(os.path.join(self.models_path, 'riskiness_model.pkl'))

            results_file = os.path.join(self.base_path, 'training_results.json')
            with open(results_file, 'r') as f:

                results = json.load(f)
                self.feature_importance = results.get('feature_importance', {})
                self.clinical_explanations = results.get('clinical_explanations', {})
            
            shap_file = os.path.join(self.models_path, 'shap_explainers.pkl')
            if os.path.exists(shap_file):
                self.explainers = joblib.load(shap_file)

                        
            print("Models loaded successfully")
            
        except FileNotFoundError as e:
            raise FileNotFoundError("Trained models not found. Please run train_model.py first.")

    def validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data"""
        required_columns = [
            'PatientID', 'Day', 'SystolicBP', 'DiastolicBP', 'HeartRate', 
            'RespirationRate', 'O2Sat', 'Glucose', 'MedicationAdherence', 
            'WeightBMI', 'ALT', 'AST', 'Bilirubin', 'Albumin', 'Creatinine', 
            'Sodium', 'Potassium', 'Cholesterol', 'SleepHours', 
            'ExerciseMinutes', 'AlcoholIntake', 'SmokingCigs'
        ]
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by PatientID and Day
        df = df.sort_values(['PatientID', 'Day']).reset_index(drop=True)
        
        # Apply outlier clipping
        outlier_bounds = {
            'SystolicBP': (50, 250), 'DiastolicBP': (30, 150),
            'HeartRate': (30, 180), 'RespirationRate': (5, 40),
            'O2Sat': (70, 100), 'WeightBMI': (10, 60),
            'Glucose': (40, 500), 'Cholesterol': (50, 400),
            'Creatinine': (0.1, 15), 'Sodium': (120, 160),
            'Potassium': (2.0, 7.0), 'Albumin': (1.0, 6.0),
            'AST': (5, 500), 'ALT': (5, 500), 'Bilirubin': (0.1, 25),
            'SleepHours': (2, 16), 'ExerciseMinutes': (0, 300),
            'AlcoholIntake': (0, 20), 'SmokingCigs': (0, 60)
        }
        
        for param, (lower, upper) in outlier_bounds.items():
            if param in df.columns:
                df[param] = np.clip(df[param], lower, upper)
        
        return df

    def calculate_slope(self, values: np.array) -> float:
        """Calculate slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope if not np.isnan(slope) else 0.0
        except:
            return 0.0

    def calculate_ema(self, values: np.array, alpha: float = 0.3) -> float:
        """Calculate Exponential Moving Average"""
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return float(ema)

    def aggregate_features(self, group_data: pd.DataFrame, feature: str, window: int, 
                          cutoff_day: int, aggregations: List[str]) -> Dict[str, float]:
        """Aggregate features for a specific window and feature"""
        
        # Filter data within the window before cutoff
        window_data = group_data[
            (group_data['Day'] >= (cutoff_day - window + 1)) & 
            (group_data['Day'] <= cutoff_day)
        ][feature].dropna()
        
        results = {}
        
        if len(window_data) == 0:
            # No data available, return zeros
            for agg in aggregations:
                results[f"{feature}_{agg}_{window}d"] = 0.0
            return results
        
        for agg in aggregations:
            feature_name = f"{feature}_{agg}_{window}d"
            
            if agg == 'mean':
                results[feature_name] = float(window_data.mean())
            elif agg == 'min':
                results[feature_name] = float(window_data.min())
            elif agg == 'max':
                results[feature_name] = float(window_data.max())
            elif agg == 'std':
                results[feature_name] = float(window_data.std()) if len(window_data) > 1 else 0.0
            elif agg == 'last':
                results[feature_name] = float(window_data.iloc[-1])
            elif agg == 'slope':
                results[feature_name] = self.calculate_slope(window_data.values)
            elif agg == 'ema':
                results[feature_name] = self.calculate_ema(window_data.values)
            elif agg == 'adherence_below_80_pct':
                # Special case for medication adherence
                pct_below_80 = (window_data < 0.8).sum() / len(window_data) * 100
                results[feature_name] = float(pct_below_80)
        
        return results

    def create_derived_features(self, patient_features: Dict[str, float]) -> Dict[str, float]:
        """Create derived clinical features"""
        derived = {}
        
        # Pulse Pressure features
        for window in [7, 14, 30, 60, 90, 180]:
            systolic_mean = patient_features.get(f'SystolicBP_mean_{window}d', 0)
            diastolic_mean = patient_features.get(f'DiastolicBP_mean_{window}d', 0)
            derived[f'PulsePressure_mean_{window}d'] = systolic_mean - diastolic_mean
            
            systolic_std = patient_features.get(f'SystolicBP_std_{window}d', 0)
            diastolic_std = patient_features.get(f'DiastolicBP_std_{window}d', 0)
            derived[f'PulsePressure_std_{window}d'] = max(systolic_std, diastolic_std)
            
            systolic_last = patient_features.get(f'SystolicBP_last_{window}d', 0)
            diastolic_last = patient_features.get(f'DiastolicBP_last_{window}d', 0)
            derived[f'PulsePressure_last_{window}d'] = systolic_last - diastolic_last

        # BMI trend
        for window in [30, 60, 90, 180]:
            bmi_slope = patient_features.get(f'WeightBMI_slope_{window}d', 0)
            derived[f'BMI_trend_{window}d'] = bmi_slope

        # Sodium/Potassium ratio
        for window in [30, 90, 180]:
            sodium_mean = patient_features.get(f'Sodium_mean_{window}d', 140)
            potassium_mean = patient_features.get(f'Potassium_mean_{window}d', 4.0)
            if potassium_mean > 0:
                derived[f'SodiumPotassiumRatio_mean_{window}d'] = sodium_mean / potassium_mean
            else:
                derived[f'SodiumPotassiumRatio_mean_{window}d'] = 0.0

        # Smoking relapse indicator
        for window in [30, 60, 180]:
            smoking_slope = patient_features.get(f'SmokingCigs_slope_{window}d', 0)
            derived[f'SmokingRelapse_{window}d'] = 1.0 if smoking_slope > 0 else 0.0

        # Combined lifestyle features
        alcohol_combined = (
            patient_features.get('AlcoholIntake_mean_30d', 0) * 0.5 +
            patient_features.get('AlcoholIntake_mean_60d', 0) * 0.3 +
            patient_features.get('AlcoholIntake_mean_180d', 0) * 0.2
        )
        derived['AlcoholIntake_combined'] = alcohol_combined
        
        smoking_combined = (
            patient_features.get('SmokingCigs_mean_30d', 0) * 0.5 +
            patient_features.get('SmokingCigs_mean_60d', 0) * 0.3 +
            patient_features.get('SmokingCigs_mean_180d', 0) * 0.2
        )
        derived['SmokingCigs_combined'] = smoking_combined

        return derived

    def create_temporal_delta_features(self, patient_features: Dict[str, float]) -> Dict[str, float]:
        """Create temporal delta/change features"""
        temporal = {}
        
        # Key features for delta calculation
        key_features = ['Glucose', 'ALT', 'AST', 'Creatinine', 'SystolicBP', 'HeartRate', 'WeightBMI']
        
        for feature in key_features:
            # Short-term vs long-term deltas
            last_7d = patient_features.get(f'{feature}_last_7d', 0)
            last_30d = patient_features.get(f'{feature}_last_30d', 0)
            last_90d = patient_features.get(f'{feature}_last_90d', 0)
            
            # Delta features (recent - baseline)
            temporal[f'{feature}_delta_7_to_30d'] = last_7d - last_30d
            temporal[f'{feature}_delta_30_to_90d'] = last_30d - last_90d
            
            # Percent change (avoid division by zero)
            if last_30d != 0:
                temporal[f'{feature}_pct_change_7_to_30d'] = ((last_7d - last_30d) / abs(last_30d)) * 100
            else:
                temporal[f'{feature}_pct_change_7_to_30d'] = 0.0
            
            if last_90d != 0:
                temporal[f'{feature}_pct_change_30_to_90d'] = ((last_30d - last_90d) / abs(last_90d)) * 100
            else:
                temporal[f'{feature}_pct_change_30_to_90d'] = 0.0

        # Trend acceleration (slope change)
        for feature in ['Glucose', 'SystolicBP', 'HeartRate']:
            slope_30d = patient_features.get(f'{feature}_slope_30d', 0)
            slope_90d = patient_features.get(f'{feature}_slope_90d', 0)
            temporal[f'{feature}_trend_acceleration'] = slope_30d - slope_90d

        return temporal

    def create_interaction_features(self, patient_features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features for clinical risk combinations"""
        interactions = {}
        
        # High-risk clinical interactions
        glucose_30d = patient_features.get('Glucose_last_30d', 100)
        bmi_30d = patient_features.get('WeightBMI_last_30d', 25)
        systolic_30d = patient_features.get('SystolicBP_last_30d', 120)
        
        # Metabolic risk interaction
        interactions['Glucose_BMI_risk'] = (glucose_30d / 100) * (bmi_30d / 25)
        interactions['BP_BMI_risk'] = (systolic_30d / 120) * (bmi_30d / 25)
        
        # Liver function interaction
        alt_30d = patient_features.get('ALT_last_30d', 25)
        ast_30d = patient_features.get('AST_last_30d', 25)
        if ast_30d > 0:
            interactions['ALT_AST_ratio'] = alt_30d / (ast_30d + 1e-3)
        else:
            interactions['ALT_AST_ratio'] = 0.0
        
        # Combined liver enzyme elevation
        interactions['Liver_enzyme_elevation'] = (alt_30d / 40) + (ast_30d / 40)
        
        # Renal-cardiac interaction
        creatinine_30d = patient_features.get('Creatinine_last_30d', 1.0)
        heart_rate_30d = patient_features.get('HeartRate_last_30d', 70)
        interactions['Renal_cardiac_risk'] = (creatinine_30d / 1.0) * (heart_rate_30d / 70)
        
        # Lifestyle combined risk score
        alcohol_30d = patient_features.get('AlcoholIntake_mean_30d', 0)
        smoking_30d = patient_features.get('SmokingCigs_mean_30d', 0)
        sleep_30d = patient_features.get('SleepHours_mean_30d', 7)
        
        # Normalized lifestyle risk (0-1 scale)
        alcohol_risk = min(alcohol_30d / 10, 1.0)
        smoking_risk = min(smoking_30d / 20, 1.0)
        sleep_risk = abs(sleep_30d - 7.5) / 7.5
        
        interactions['Lifestyle_risk_score'] = (alcohol_risk + smoking_risk + sleep_risk) / 3
        
        return interactions

    def engineer_features_for_patient(self, patient_data: pd.DataFrame) -> Dict[str, float]:
        """Engineer features for a single patient"""
        
        # Get the last recorded day as cutoff
        cutoff_day = patient_data['Day'].max()
        
        patient_features = {'PatientID': patient_data['PatientID'].iloc[0]}
        
        # Process each feature group
        for group_name, config in self.feature_config.items():
            for feature in config['features']:
                for window in config['windows']:
                    feature_results = self.aggregate_features(
                        patient_data, feature, window, cutoff_day, config['aggregations']
                    )
                    patient_features.update(feature_results)

        # Add derived features
        derived_features = self.create_derived_features(patient_features)
        patient_features.update(derived_features)
        
        # Add temporal delta features
        temporal_features = self.create_temporal_delta_features(patient_features)
        patient_features.update(temporal_features)
        
        # Add interaction features
        interaction_features = self.create_interaction_features(patient_features)
        patient_features.update(interaction_features)
        
        return patient_features

    def infer_chronic_conditions(self, patient_data: pd.DataFrame) -> List[str]:
        """Infer likely chronic conditions based on clinical patterns"""
        conditions = []
        
        # Get recent values (last 30 days)
        recent_data = patient_data.tail(min(30, len(patient_data)))
        
        # Diabetes indicators
        avg_glucose = recent_data['Glucose'].mean()
        if avg_glucose > 126:  # Diabetic range
            conditions.append('Diabetes')
        
        # Hypertension
        avg_systolic = recent_data['SystolicBP'].mean()
        avg_diastolic = recent_data['DiastolicBP'].mean()
        if avg_systolic > 140 or avg_diastolic > 90:
            conditions.append('Hypertension')
        
        # Obesity
        avg_bmi = recent_data['WeightBMI'].mean()
        if avg_bmi > 30:
            conditions.append('Obesity')
        
        # CKD (Chronic Kidney Disease)
        avg_creatinine = recent_data['Creatinine'].mean()
        if avg_creatinine > 1.5:
            conditions.append('CKD')
        
        # Liver Disease
        avg_alt = recent_data['ALT'].mean()
        avg_ast = recent_data['AST'].mean()
        if avg_alt > 80 or avg_ast > 80:
            conditions.append('Liver_Disease')
        
        # Heart conditions (basic indicators)
        avg_hr = recent_data['HeartRate'].mean()
        if avg_hr > 100 and avg_systolic > 140:
            conditions.append('Heart_Failure')
        
        # Respiratory issues
        avg_o2sat = recent_data['O2Sat'].mean()
        avg_resp_rate = recent_data['RespirationRate'].mean()
        if avg_o2sat < 92 or avg_resp_rate > 24:
            conditions.append('Asthma')
        
        return conditions if conditions else ['None']

    def get_risk_level(self, risk_score: float) -> str:
        """Convert numeric risk score to risk level"""
        if risk_score < 0.25:
            return 'Low'
        elif risk_score < 0.5:
            return 'Medium-Low'
        elif risk_score < 0.75:
            return 'Medium-High'
        else:
            return 'High'

    def get_local_explanations(self, patient_features: pd.DataFrame, patient_id: int) -> Dict[str, Any]:
        """Get local SHAP explanations for specific patient"""
        explanations = {
            'deterioration': {},
            'riskiness': {}
        }
        
        if self.explainers:
            try:
                # Get SHAP values for this patient
                for model_type in ['deterioration', 'riskiness']:
                    if model_type in self.explainers:
                        explainer = self.explainers[model_type]['explainer']
                        shap_values = explainer.shap_values(patient_features)
                        
                        # Get top contributing features
                        feature_names = patient_features.columns
                        shap_importance = dict(zip(feature_names, np.abs(shap_values[0])))
                        
                        # Sort and get top 5
                        top_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        
                        explanations[model_type] = {
                            'top_features': top_features,
                            'base_value': explainer.expected_value,
                            'prediction_contribution': dict(zip(feature_names, shap_values[0]))
                        }
            except Exception as e:
                print(f"Could not generate local explanations: {e}")
        
        return explanations

    def make_predictions(self, input_file: str) -> pd.DataFrame:
        """Make predictions on new patient data"""
        print(f"Making predictions on {input_file}...")
        
        # Load and validate input data
        df = pd.read_csv(input_file)
        df = self.validate_input_data(df)
        
        predictions = []
        
        total_patients = df['PatientID'].nunique()
        processed = 0
        
        for patient_id, patient_data in df.groupby('PatientID'):
            # Engineer features for this patient
            patient_features = self.engineer_features_for_patient(patient_data)
            
            # Convert to DataFrame format expected by model
            feature_df = pd.DataFrame([patient_features])
            
            # Remove PatientID for model input
            model_input = feature_df.drop(['PatientID'], axis=1, errors='ignore')
            
            # Ensure all expected features are present (fill missing with 0)
            if hasattr(self.deterioration_model, 'feature_names_in_'):
                expected_features = self.deterioration_model.feature_names_in_
                for feature in expected_features:
                    if feature not in model_input.columns:
                        model_input[feature] = 0.0
                model_input = model_input[expected_features]
            
            # Make predictions
            deterioration_prob = self.deterioration_model.predict_proba(model_input)[0, 1]
            deterioration_pred = int(deterioration_prob >= 0.5)
            
            risk_score = np.clip(self.riskiness_model.predict(model_input)[0], 0, 1)
            risk_level = self.get_risk_level(risk_score)
            
            # Infer chronic conditions
            chronic_conditions = self.infer_chronic_conditions(patient_data)
            
            # Get local explanations
            local_explanations = self.get_local_explanations(model_input, patient_id)
            
            # Compile prediction result
            prediction = {
                'PatientID': patient_id,
                'Risk_Score': float(risk_score),
                'Risk_Level': risk_level,
                'Deterioration_Probability': float(deterioration_prob),
                'Deterioration_Prediction': deterioration_pred,
                'Predicted_Chronic_Conditions': '|'.join(chronic_conditions),
                'Days_of_Data': len(patient_data),
                'Last_Recorded_Day': int(patient_data['Day'].max()),
                'Top_Risk_Factors': self.get_top_risk_factors(patient_features),
                'Clinical_Alerts': self.generate_clinical_alerts(patient_data, risk_score)
            }
            
            predictions.append(prediction)
            processed += 1
            
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_patients} patients")
        
        return pd.DataFrame(predictions)

    def get_top_risk_factors(self, patient_features: Dict[str, float]) -> str:
        """Identify top risk factors contributing to patient risk"""
        risk_factors = []
        
        # Check for high-risk values
        glucose_30d = patient_features.get('Glucose_last_30d', 100)
        if glucose_30d > 180:
            risk_factors.append('High_Glucose')
        
        systolic_30d = patient_features.get('SystolicBP_last_30d', 120)
        if systolic_30d > 160:
            risk_factors.append('Severe_Hypertension')
        
        creatinine_30d = patient_features.get('Creatinine_last_30d', 1.0)
        if creatinine_30d > 2.0:
            risk_factors.append('Kidney_Dysfunction')
        
        o2sat_30d = patient_features.get('O2Sat_last_30d', 98)
        if o2sat_30d < 90:
            risk_factors.append('Low_Oxygen')
        
        med_adherence = patient_features.get('MedicationAdherence_last_30d', 95)
        if med_adherence < 70:
            risk_factors.append('Poor_Medication_Adherence')
        
        alt_30d = patient_features.get('ALT_last_30d', 25)
        if alt_30d > 80:
            risk_factors.append('Liver_Dysfunction')
        
        bmi_30d = patient_features.get('WeightBMI_last_30d', 25)
        if bmi_30d > 35:
            risk_factors.append('Severe_Obesity')
        
        return '|'.join(risk_factors) if risk_factors else 'None_Identified'

    def generate_clinical_alerts(self, patient_data: pd.DataFrame, risk_score: float) -> str:
        """Generate clinical alerts based on patient data"""
        alerts = []
        
        recent_data = patient_data.tail(7)  # Last week of data
        
        # Critical value alerts
        if recent_data['O2Sat'].min() < 88:
            alerts.append('CRITICAL_LOW_OXYGEN')
        
        if recent_data['Glucose'].max() > 300:
            alerts.append('CRITICAL_HIGH_GLUCOSE')
        
        if recent_data['SystolicBP'].max() > 180:
            alerts.append('HYPERTENSIVE_CRISIS')
        
        if recent_data['HeartRate'].max() > 140:
            alerts.append('SEVERE_TACHYCARDIA')
        
        # Trend alerts
        if len(patient_data) >= 14:
            glucose_trend = patient_data.tail(14)['Glucose'].rolling(7).mean()
            if len(glucose_trend.dropna()) >= 2:
                if glucose_trend.iloc[-1] > glucose_trend.iloc[-2] * 1.2:
                    alerts.append('RAPIDLY_RISING_GLUCOSE')
        
        # Risk-based alerts
        if risk_score > 0.8:
            alerts.append('HIGH_DETERIORATION_RISK')
        
        # Medication adherence alert
        if recent_data['MedicationAdherence'].mean() < 60:
            alerts.append('POOR_MEDICATION_COMPLIANCE')
        
        return '|'.join(alerts) if alerts else 'No_Alerts'

    def create_patient_report(self, predictions_df: pd.DataFrame, patient_id: int) -> Dict[str, Any]:
        """Create detailed report for a specific patient"""
        patient_pred = predictions_df[predictions_df['PatientID'] == patient_id].iloc[0]
        
        # Clinical interpretation
        risk_interpretation = {
            'Low': 'Patient shows minimal signs of clinical deterioration risk. Continue routine monitoring.',
            'Medium-Low': 'Patient has some risk factors present. Increase monitoring frequency and address modifiable risks.',
            'Medium-High': 'Patient has multiple risk factors. Consider more intensive interventions and closer monitoring.',
            'High': 'Patient is at high risk for deterioration. Immediate clinical review recommended with possible intervention.'
        }
        
        # Risk factor explanations
        top_factors = patient_pred['Top_Risk_Factors'].split('|') if patient_pred['Top_Risk_Factors'] != 'None_Identified' else []
        factor_explanations = {
            'High_Glucose': 'Blood glucose levels indicate poor diabetic control or new onset diabetes',
            'Severe_Hypertension': 'Blood pressure readings suggest severe hypertension requiring immediate attention',
            'Kidney_Dysfunction': 'Elevated creatinine indicates reduced kidney function',
            'Low_Oxygen': 'Oxygen saturation below normal range suggests respiratory compromise',
            'Poor_Medication_Adherence': 'Patient not taking medications as prescribed',
            'Liver_Dysfunction': 'Elevated liver enzymes indicate potential liver damage or disease',
            'Severe_Obesity': 'BMI indicates severe obesity with associated health risks'
        }
        
        report = {
            'patient_id': int(patient_id),
            'overall_risk_assessment': {
                'risk_score': float(patient_pred['Risk_Score']),
                'risk_level': patient_pred['Risk_Level'],
                'deterioration_probability': float(patient_pred['Deterioration_Probability']),
                'clinical_interpretation': risk_interpretation.get(patient_pred['Risk_Level'], 'Unknown risk level')
            },
            'predicted_conditions': {
                'conditions': patient_pred['Predicted_Chronic_Conditions'].split('|') if patient_pred['Predicted_Chronic_Conditions'] != 'None' else [],
                'confidence': 'Based on clinical patterns in patient data'
            },
            'key_risk_factors': [
                {
                    'factor': factor,
                    'explanation': factor_explanations.get(factor, 'Refer to clinical guidelines'),
                    'priority': 'High' if factor in ['CRITICAL_LOW_OXYGEN', 'CRITICAL_HIGH_GLUCOSE', 'HYPERTENSIVE_CRISIS'] else 'Medium'
                }
                for factor in top_factors
            ],
            'clinical_alerts': {
                'alerts': patient_pred['Clinical_Alerts'].split('|') if patient_pred['Clinical_Alerts'] != 'No_Alerts' else [],
                'requires_immediate_attention': any(alert.startswith('CRITICAL_') for alert in patient_pred['Clinical_Alerts'].split('|'))
            },
            'data_quality': {
                'days_of_data': int(patient_pred['Days_of_Data']),
                'data_completeness': 'Adequate' if patient_pred['Days_of_Data'] >= 30 else 'Limited',
                'last_recorded_day': int(patient_pred['Last_Recorded_Day'])
            },
            'recommendations': self.get_clinical_recommendations(patient_pred)
        }
        
        return report

    def get_clinical_recommendations(self, patient_pred: pd.Series) -> List[str]:
        """Generate clinical recommendations based on predictions"""
        recommendations = []
        
        risk_level = patient_pred['Risk_Level']
        alerts = patient_pred['Clinical_Alerts'].split('|') if patient_pred['Clinical_Alerts'] != 'No_Alerts' else []
        conditions = patient_pred['Predicted_Chronic_Conditions'].split('|') if patient_pred['Predicted_Chronic_Conditions'] != 'None' else []
        
        # Risk-based recommendations
        if risk_level == 'High':
            recommendations.append('Schedule immediate clinical review within 24-48 hours')
            recommendations.append('Consider inpatient monitoring if clinically indicated')
        elif risk_level == 'Medium-High':
            recommendations.append('Increase monitoring frequency to weekly visits')
            recommendations.append('Review and optimize current treatment plan')
        elif risk_level == 'Medium-Low':
            recommendations.append('Schedule follow-up within 2 weeks')
            recommendations.append('Address modifiable risk factors')
        
        # Condition-specific recommendations
        if 'Diabetes' in conditions:
            recommendations.append('Optimize diabetes management - consider endocrinology referral')
        if 'Hypertension' in conditions:
            recommendations.append('Review antihypertensive medications and lifestyle modifications')
        if 'CKD' in conditions:
            recommendations.append('Nephrology consultation recommended')
        if 'Heart_Failure' in conditions:
            recommendations.append('Cardiology evaluation and heart failure management optimization')
        
        # Alert-based recommendations
        if 'POOR_MEDICATION_COMPLIANCE' in alerts:
            recommendations.append('Medication adherence counseling and support')
        if 'HIGH_DETERIORATION_RISK' in alerts:
            recommendations.append('Consider care coordination and case management')
        
        # General recommendations
        if patient_pred['Days_of_Data'] < 30:
            recommendations.append('Establish consistent monitoring routine for better risk assessment')
        
        return recommendations

    def save_predictions(self, predictions_df: pd.DataFrame, output_file: str = None):
        """Save predictions and generate summary reports"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"patient_predictions_{timestamp}.csv"
        
        # Save main predictions
        predictions_df.to_csv(output_file, index=False)
        
        # Create summary statistics
        summary_stats = {
            'total_patients': len(predictions_df),
            'risk_distribution': predictions_df['Risk_Level'].value_counts().to_dict(),
            'deterioration_predictions': {
                'high_risk': (predictions_df['Deterioration_Prediction'] == 1).sum(),
                'low_risk': (predictions_df['Deterioration_Prediction'] == 0).sum(),
                'average_probability': float(predictions_df['Deterioration_Probability'].mean())
            },
            'common_conditions': {},
            'alert_summary': {}
        }
        
        # Analyze common conditions
        all_conditions = []
        for conditions_str in predictions_df['Predicted_Chronic_Conditions']:
            if conditions_str != 'None':
                all_conditions.extend(conditions_str.split('|'))
        
        from collections import Counter
        condition_counts = Counter(all_conditions)
        summary_stats['common_conditions'] = dict(condition_counts.most_common(10))
        
        # Analyze alerts
        all_alerts = []
        for alerts_str in predictions_df['Clinical_Alerts']:
            if alerts_str != 'No_Alerts':
                all_alerts.extend(alerts_str.split('|'))
        
        alert_counts = Counter(all_alerts)
        summary_stats['alert_summary'] = dict(alert_counts.most_common(10))

        # Save summary
        summary_file = output_file.replace('.csv', '_summary.json')

        # helper function to convert numpy types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            else:
                return obj

        # now actually save the summary using that function
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=convert_to_serializable)

        # Generate individual patient reports for high-risk patients
        high_risk_patients = predictions_df[predictions_df['Risk_Level'] == 'High']['PatientID'].tolist()

        if high_risk_patients:
            reports_dir = 'high_risk_patient_reports'
            os.makedirs(self.reports_dir, exist_ok=True)
            
            for patient_id in high_risk_patients[:10]:  # Limit to first 10 high-risk patients
                patient_report = self.create_patient_report(predictions_df, patient_id)
                report_file = os.path.join(self.reports_dir, f'patient_{patient_id}_report.json')
                with open(report_file, 'w') as f:
                    json.dump(patient_report, f, indent=2)

        print(f"Predictions saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")
        if high_risk_patients:
            print(f"High-risk patient reports saved to: {reports_dir}/")
        
        return output_file, summary_stats

    def print_prediction_summary(self, predictions_df: pd.DataFrame, summary_stats: Dict):
        """Print summary of predictions"""
        print("\n" + "="*80)
        print("HEALTHCARE AI PREDICTION SUMMARY")
        print("="*80)
        
        print(f"Total patients processed: {summary_stats['total_patients']:,}")
        
        print(f"\nRisk Level Distribution:")
        for level, count in summary_stats['risk_distribution'].items():
            pct = (count / summary_stats['total_patients']) * 100
            print(f"  {level}: {count:,} patients ({pct:.1f}%)")
        
        print(f"\nDeterioration Risk Predictions:")
        det_stats = summary_stats['deterioration_predictions']
        print(f"  High risk (>50%): {det_stats['high_risk']:,} patients")
        print(f"  Low risk (≤50%): {det_stats['low_risk']:,} patients")
        print(f"  Average probability: {det_stats['average_probability']:.3f}")
        
        print(f"\nMost Common Predicted Conditions:")
        for condition, count in list(summary_stats['common_conditions'].items())[:5]:
            print(f"  {condition}: {count:,} patients")
        
        print(f"\nMost Common Clinical Alerts:")
        for alert, count in list(summary_stats['alert_summary'].items())[:5]:
            print(f"  {alert}: {count:,} patients")
        
        # Highlight critical cases
        critical_cases = predictions_df[predictions_df['Clinical_Alerts'].str.contains('CRITICAL_', na=False)]
        if len(critical_cases) > 0:
            print(f"\n⚠️  CRITICAL ALERTS: {len(critical_cases)} patients require immediate attention")
        
        high_risk_count = len(predictions_df[predictions_df['Risk_Level'] == 'High'])
        if high_risk_count > 0:
            print(f"🔴 HIGH RISK: {high_risk_count} patients need clinical review within 24-48 hours")

def main():
    """Main prediction function"""
    print("🏥 Healthcare AI Prediction System")
    print("="*50)
    
    predictor = HealthcarePredictor()
    
    try:
        # Load trained models
        predictor.load_trained_models()
        
        # Make predictions (assumes input file is 'patient_data.csv')
        input_file = os.path.join(predictor.base_path, 'predict.csv')
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            print("Please provide a CSV file with patient data containing the required columns:")
            print("PatientID,Day,SystolicBP,DiastolicBP,HeartRate,RespirationRate,O2Sat,Glucose,MedicationAdherence,WeightBMI,ALT,AST,Bilirubin,Albumin,Creatinine,Sodium,Potassium,Cholesterol,SleepHours,ExerciseMinutes,AlcoholIntake,SmokingCigs")
            return
        
        predictions_df = predictor.make_predictions(input_file)
        
        # Save predictions and generate reports
        output_file, summary_stats = predictor.save_predictions(predictions_df)
        
        # Print summary
        predictor.print_prediction_summary(predictions_df, summary_stats)
        
        print(f"\n✅ Prediction pipeline completed successfully!")
        print("\nGenerated files:")
        print(f"  - {output_file} (main predictions)")
        print(f"  - {output_file.replace('.csv', '_summary.json')} (summary statistics)")
        print("  - high_risk_patient_reports/ (individual reports for high-risk patients)")

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()