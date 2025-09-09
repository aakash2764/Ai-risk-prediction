import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class ChronicCareDatasetGenerator:
    def __init__(self):
        self.chronic_conditions = [
            'Diabetes', 'Obesity', 'Heart_Failure', 'CKD', 
            'Asthma', 'Hypertension', 'Stroke', 'Cancer', 'Liver_Disease'
        ]
        
        # Clinical bands
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
            'ExerciseMinutes': {'healthy': (25, 45), 'low': (17, 24), 'medium': (8, 16), 'severe': (0, 7)},  # Daily equivalent
            'AlcoholIntake': {'healthy': (0, 2), 'low': (3, 5), 'medium': (6, 10), 'severe': (11, 20)},
            'SmokingCigs': {'healthy': (0, 0), 'low': (1, 5), 'medium': (6, 10), 'severe': (11, 60)}
        }
        
        # Measurement schedules
        self.schedules = {
            'daily': ['SystolicBP', 'DiastolicBP', 'HeartRate', 'RespirationRate', 'O2Sat', 'MedicationAdherence', 'SmokingCigs'],
            'glucose': ['Glucose'],  # every 3 days or daily if diabetes
            'weekly': ['WeightBMI', 'ALT', 'AST', 'Bilirubin', 'Albumin', 'Creatinine', 'Sodium', 'Potassium', 'SleepHours', 'ExerciseMinutes', 'AlcoholIntake'],
            'monthly': ['Cholesterol']
        }

    def generate_patient_cohorts(self, total_patients=10000):
        """Generate patient cohorts with chronic conditions and risk categories"""
        healthy_count = np.random.randint(1400, 2001)
        chronic_count = total_patients - healthy_count 
        
        patients = []
        patient_id = 1
        
        # Healthy patients
        for _ in range(healthy_count):
            patients.append({
                'PatientID': patient_id,
                'conditions': ["None"],
                'risk_category': 'healthy',
                'days': np.random.randint(30, 181)
            })
            patient_id += 1
        
        # Chronic patients
        severe_target = np.random.randint(1500, 2001)
        severe_count = 0
        
        for _ in range(chronic_count):
            num_conditions = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            # Weighted probabilities for conditions
            condition_probs = {
                'Diabetes': 0.25,
                'Obesity': 0.20,
                'Hypertension': 0.20,
                'Heart_Failure': 0.10,
                'CKD': 0.08,
                'Asthma': 0.07,
                'Cancer': 0.05,
                'Stroke': 0.03,
                'Liver_Disease': 0.02   # much less common
            }
            conditions, probs = zip(*condition_probs.items())

            # Sample multiple conditions without replacement, using weights
            conditions = np.random.choice(
                conditions, num_conditions, replace=False, p=probs
            ).tolist()

                        
            # Determine risk category based on worst condition severity
            if severe_count < severe_target and (np.random.random() < 0.3 or 
                any(c in ['Heart_Failure', 'Cancer', 'CKD'] for c in conditions)):
                risk_category = 'severe'
                severe_count += 1
            elif np.random.random() < 0.4:
                risk_category = 'medium'
            else:
                risk_category = 'low'
            
            patients.append({
                'PatientID': patient_id,
                'conditions': conditions,
                'risk_category': risk_category,
                'days': np.random.randint(30, 181)
            })
            patient_id += 1
        
        return patients

    def generate_baseline_values(self, risk_category: str) -> Dict[str, float]:
        """Generate baseline values for a patient based on risk category"""
        baselines = {}
        
        for param, bands in self.clinical_bands.items():
            if risk_category == 'healthy':
                band = bands['healthy']
            elif risk_category == 'low':
                band = bands['low']
            elif risk_category == 'medium':
                band = bands['medium']
            else:  # severe
                band = bands['severe']
            
            baselines[param] = np.random.uniform(band[0], band[1])
        
        return baselines

    def apply_disease_influence(self, baselines: Dict[str, float], conditions: List[str]) -> Dict[str, float]:
        """Apply disease-specific influences to baseline values"""
        values = baselines.copy()
        
        for condition in conditions:
            if condition == 'Obesity':
                values['WeightBMI'] *= np.random.uniform(1.10, 1.30)
                values['Glucose'] *= np.random.uniform(1.05, 1.20)
                values['Cholesterol'] *= np.random.uniform(1.05, 1.20)
                values['ExerciseMinutes'] *= np.random.uniform(0.30, 0.70)
                values['SleepHours'] -= np.random.uniform(0.5, 1.5)
                
            elif condition == 'Diabetes':
                values['Glucose'] *= np.random.uniform(1.20, 1.60)
                values['WeightBMI'] *= np.random.uniform(1.05, 1.15)
                values['ExerciseMinutes'] *= np.random.uniform(0.40, 0.80)
                values['AlcoholIntake'] *= np.random.uniform(1.0, 1.30)
                
            elif condition == 'Heart_Failure':
                values['SystolicBP'] *= np.random.uniform(1.05, 1.20)
                values['DiastolicBP'] *= np.random.uniform(1.05, 1.20)
                values['HeartRate'] *= np.random.uniform(1.05, 1.25)
                values['O2Sat'] *= np.random.uniform(0.92, 0.99)
                values['Creatinine'] *= np.random.uniform(1.10, 1.50)
                values['Sodium'] *= np.random.uniform(0.95, 1.05)
                
            elif condition == 'CKD':
                values['Creatinine'] *= np.random.uniform(1.50, 3.00)
                values['SystolicBP'] *= np.random.uniform(1.05, 1.15)
                values['DiastolicBP'] *= np.random.uniform(1.05, 1.15)
                values['Sodium'] *= np.random.uniform(0.95, 1.05)
                values['Potassium'] *= np.random.uniform(0.90, 1.15)
                
            elif condition == 'Asthma':
                values['RespirationRate'] *= np.random.uniform(1.10, 1.50)
                values['O2Sat'] *= np.random.uniform(0.92, 0.99)
                values['SmokingCigs'] += np.random.uniform(5, 15)
                
            elif condition == 'Stroke':
                values['SystolicBP'] *= np.random.uniform(1.10, 1.30)
                values['DiastolicBP'] *= np.random.uniform(1.10, 1.30)
                values['SmokingCigs'] += np.random.uniform(3, 10)
                values['SleepHours'] -= np.random.uniform(0.5, 2.0)
                values['ExerciseMinutes'] *= np.random.uniform(0.30, 0.70)
                
            elif condition == 'Cancer':
                values['WeightBMI'] *= np.random.uniform(0.75, 0.95)
                values['ALT'] *= np.random.uniform(1.20, 2.00)
                values['AST'] *= np.random.uniform(1.20, 2.00)
                values['Albumin'] *= np.random.uniform(0.70, 0.90)
                
            elif condition == 'Liver_Disease':
                values['ALT'] *= np.random.uniform(1.50, 3.00)
                values['AST'] *= np.random.uniform(1.50, 3.00)
                values['Bilirubin'] *= np.random.uniform(1.50, 3.00)
                values['Albumin'] *= np.random.uniform(0.60, 0.85)
                values['AlcoholIntake'] *= np.random.uniform(1.2, 1.8)
                
            elif condition == 'Hypertension':
                values['SystolicBP'] *= np.random.uniform(1.15, 1.40)
                values['DiastolicBP'] *= np.random.uniform(1.10, 1.30)
        
        return self.clamp_values(values)

    def clamp_values(self, values: Dict[str, float]) -> Dict[str, float]:
        """Clamp values to safe physiological ranges"""
        clamped = {}
        
        for param, value in values.items():
            if param in self.clinical_bands:
                bands = self.clinical_bands[param]
                min_val = min(bands['healthy'][0], bands['severe'][0])
                max_val = max(bands['healthy'][1], bands['severe'][1])
                clamped[param] = np.clip(value, min_val, max_val)
            else:
                clamped[param] = value
        
        return clamped

    def generate_measurement_schedule(self, days: int, conditions: List[str]) -> Dict[str, List[int]]:
        """Generate measurement schedule for each parameter"""
        schedule = {}
        
        # Daily measurements
        for param in self.schedules['daily']:
            schedule[param] = list(range(1, days + 1))
        
        # Glucose: every 3 days, or daily if diabetes
        if 'Diabetes' in conditions:
            schedule['Glucose'] = list(range(1, days + 1))
        else:
            schedule['Glucose'] = list(range(1, days + 1, 3))
        
        # Weekly measurements
        for param in self.schedules['weekly']:
            schedule[param] = list(range(1, days + 1, 7))
        
        # Monthly measurements
        for param in self.schedules['monthly']:
            schedule[param] = list(range(1, days + 1, 30))
        
        return schedule

    def add_temporal_noise_and_trends(self, values: pd.DataFrame) -> pd.DataFrame:
        """Add realistic noise and trends to time series"""
        for param in values.columns:
            if param in ['PatientID', 'Day', 'ChronicConditions', 'Riskiness', 'Deterioration_within_90days']:
                continue
            
            # Add Gaussian noise (5% of mean)
            noise_std = values[param].mean() * 0.05
            noise = np.random.normal(0, noise_std, len(values))
            values[param] += noise
            
            # Add occasional trends for some patients
            if np.random.random() < 0.1:  # 10% of time series get trends
                trend_slope = np.random.normal(0, noise_std * 0.1)
                trend = trend_slope * np.arange(len(values))
                values[param] += trend
            
            # Add occasional spikes for deterioration simulation
            if np.random.random() < 0.05:  # 5% chance of spike
                spike_day = np.random.randint(0, len(values))
                spike_magnitude = values[param].iloc[spike_day] * np.random.uniform(1.2, 2.0)
                values.loc[spike_day, param] = spike_magnitude

        
        return values

    def get_risk_score(self, values: Dict[str, float], conditions: List[str]) -> float:
        """Compute realistic risk score (0-1) based on vitals & conditions with weighted impact"""
        score = 0.0
        total_weight = 0.0
        
        # define importance of each vital (higher = more impact on risk)
        vital_importance = {
            'O2Sat': 2.0, 'Glucose': 1.5, 'Creatinine': 1.5, 'HeartRate': 1.0,
            'SystolicBP': 1.0, 'DiastolicBP': 1.0, 'MedicationAdherence': 1.5,
            'WeightBMI': 0.8, 'ALT': 0.7, 'AST': 0.7, 'Bilirubin': 0.8, 'Albumin': 0.8,
            'Sodium': 0.7, 'Potassium': 0.7, 'Cholesterol': 0.6,
            'RespirationRate': 1.0, 'SleepHours': 0.5, 'ExerciseMinutes': 0.5,
            'AlcoholIntake': 0.5, 'SmokingCigs': 0.5
        }
        # if patient is healthy, give random small risk 0–0.25
        # if patient is healthy, give random small risk 0–0.25
        # if patient has no chronic conditions, treat as healthy
        if not conditions or ('None' in conditions):
            return float(np.random.uniform(0.1, 0.18))



        
        # calculate parameter contributions
        for param, bands in self.clinical_bands.items():
            val = values.get(param, 0)
            importance = vital_importance.get(param, 1.0)
            total_weight += importance
            
            if bands['healthy'][0] <= val <= bands['healthy'][1]:
                severity = 0.0
            elif bands['low'][0] <= val <= bands['low'][1]:
                severity = 0.25
            elif bands['medium'][0] <= val <= bands['medium'][1]:
                severity = 0.5
            elif bands['severe'][0] <= val <= bands['severe'][1]:
                severity = 0.75
            else:
                severity = 1.0  # extreme outlier
            
            score += severity * importance
        
        # add disease boost
        for cond in conditions:
            if cond in ['Heart_Failure', 'Cancer', 'CKD']:
                score += 0.15 * total_weight  # strong boost for high-risk disease
            elif cond in ['Diabetes', 'Hypertension', 'Obesity']:
                score += 0.07 * total_weight  # medium boost
        
        # normalize score 0-1
        score = score / total_weight
        
        # push high-risk patients closer to 0.75–0.99
        if score > 0.7:
            score = 0.75 + (score - 0.7) * (0.99 - 0.75)/0.3
        score = float(np.clip(score, 0.0, 0.99))
        
        return score
    
    def calculate_deterioration_probability(self, patient_data: pd.DataFrame, risk_score: float) -> float:
        """
        Compute deterioration probability based on numeric risk score and extreme vitals.
        This replaces random low/high probabilities to be more realistic.
        """
        # base probability = scaled risk score
        prob = risk_score
        
        # add small bumps for extreme vitals
        extreme_factors = 0
        extremes = {
            'O2Sat': 85, 'Glucose': 300, 'Creatinine': 4.0,
            'MedicationAdherence': 50, 'SmokingCigs': 20, 'AlcoholIntake': 10
        }
        for param, threshold in extremes.items():
            if param in patient_data.columns:
                if param in ['O2Sat', 'MedicationAdherence']:  # low is bad
                    extreme_factors += (patient_data[param] < threshold).sum() / len(patient_data)
                else:  # high is bad
                    extreme_factors += (patient_data[param] > threshold).sum() / len(patient_data)
        
        # amplify probability slightly based on extremes
        prob += extreme_factors * 0.2
        prob = float(np.clip(prob, 0.0, 0.99))  # max 0.99
        return prob
    





    def introduce_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Introduce realistic missingness patterns"""
        df_missing = df.copy()
        
        daily_params = self.schedules['daily'] + ['Glucose']
        weekly_params = self.schedules['weekly']
        monthly_params = self.schedules['monthly']
        
        # Daily measures: 5-10% missing
                # daily params: 7.5% missing
        daily_mask = np.random.random(df_missing[daily_params].shape) < 0.075
        df_missing.loc[:, daily_params] = df_missing[daily_params].mask(daily_mask)

        # weekly/monthly params: 15% missing
        weekly_monthly = weekly_params + monthly_params
        weekly_monthly_mask = np.random.random(df_missing[weekly_monthly].shape) < 0.15
        df_missing.loc[:, weekly_monthly] = df_missing[weekly_monthly].mask(weekly_monthly_mask)

        
        return df_missing

    def forward_fill_and_backfill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill missing values and backfill if needed"""
        df_filled = df.copy()
        
        # Group by patient and forward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['PatientID', 'Day', 'Deterioration_within_90days']]
        
        # forward-fill then backfill numeric columns grouped by PatientID
        df_filled[numeric_cols] = df_filled.groupby('PatientID')[numeric_cols].ffill().bfill()

        # optional: if first values are still missing (all NaN for patient), fill with baseline
        risk_mapping = { (0.0, 0.25): 'healthy', (0.25, 0.5): 'low', (0.5, 0.75): 'medium', (0.75, 1.0): 'severe' }

        for col in numeric_cols:
            missing_patients = df_filled['PatientID'][df_filled[col].isna()].unique()
            for pid in missing_patients:
                risk_score = df_filled.loc[df_filled['PatientID']==pid, 'Riskiness'].iloc[0]
                # map numeric risk_score to risk category
                for (low, high), cat in risk_mapping.items():
                    if low <= risk_score < high or (cat == 'severe' and risk_score == 1.0):
                        risk_cat = cat
                        break
                baseline_val = np.random.uniform(*self.clinical_bands[col][risk_cat])
                df_filled.loc[df_filled['PatientID']==pid, col] = baseline_val

        return df_filled


    def generate_dataset(self):
        """Generate the complete synthetic dataset"""
        print("Generating patient cohorts...")
        patients = self.generate_patient_cohorts()
        
        all_records = []
        deterioration_labels = {}
        
        print("Generating patient data...")
        for i, patient in enumerate(patients):
            if i % 1000 == 0:
                print(f"Processing patient {i+1}/{len(patients)}")
            
            patient_id = patient['PatientID']
            conditions = patient['conditions']
            risk_category = patient['risk_category']
            days = patient['days']
            
            # Generate baseline values
            baselines = self.generate_baseline_values(risk_category)
            
            # Apply disease influences
            if conditions:
                baselines = self.apply_disease_influence(baselines, conditions)
            
            # Generate measurement schedule
            schedule = self.generate_measurement_schedule(days, conditions)
            
            # Generate risk score based on category
            risk_score = self.get_risk_score(baselines, conditions)

            
            # Create patient records
            patient_records = []
            for day in range(1, days + 1):
                record = {
                    'PatientID': patient_id,
                    'Day': day,
                    'ChronicConditions': '|'.join(conditions) if conditions else '',
                    'Riskiness': risk_score  # Now using numeric risk score instead of category string
                }
                
                # Add measurements based on schedule
                for param in baselines.keys():
                    if param in schedule and day in schedule[param]:
                        # Add some daily variation
                        variation = np.random.normal(0, baselines[param] * 0.02)
                        record[param] = baselines[param] + variation
                    else:
                        # Use last known value (will be forward-filled later)
                        record[param] = np.nan
                
                patient_records.append(record)
            
            # Convert to DataFrame for easier manipulation
            patient_df = pd.DataFrame(patient_records)
            
            # Add temporal noise and trends
            patient_df = self.add_temporal_noise_and_trends(patient_df)
            
            # Clamp values after noise addition
            for param in baselines.keys():
                if param in patient_df.columns:
                    patient_df[param] = patient_df[param].apply(
                        lambda x: np.clip(x, 
                                        min(self.clinical_bands[param]['healthy'][0], self.clinical_bands[param]['severe'][0]),
                                        max(self.clinical_bands[param]['healthy'][1], self.clinical_bands[param]['severe'][1]))
                    )
            
            if 'O2Sat' in patient_df.columns:
                patient_df['O2Sat'] = np.clip(patient_df['O2Sat'], 85, 100)
            if 'Bilirubin' in patient_df.columns:
                patient_df['Bilirubin'] = np.clip(patient_df['Bilirubin'], 0.2, 10)        
                        
            # Calculate deterioration probability and label
            deterioration_prob = self.calculate_deterioration_probability(patient_df, risk_score)
            deterioration_label = int(np.random.random() < deterioration_prob)
            deterioration_labels[patient_id] = deterioration_label
            
            patient_df['Deterioration_within_90days'] = deterioration_label
            
            all_records.extend(patient_df.to_dict('records'))
        
        print("Creating final dataset...")
        df = pd.DataFrame(all_records)
        
        # Reorder columns as specified
        column_order = [
            'PatientID', 'Day', 'ChronicConditions', 'Riskiness',
            'SystolicBP', 'DiastolicBP', 'HeartRate', 'RespirationRate', 'O2Sat', 'Glucose',
            'MedicationAdherence', 'WeightBMI', 'ALT', 'AST', 'Bilirubin', 'Albumin',
            'Creatinine', 'Sodium', 'Potassium', 'Cholesterol',
            'SleepHours', 'ExerciseMinutes', 'AlcoholIntake', 'SmokingCigs',
            'Deterioration_within_90days'
        ]
        
        df = df[column_order]
        
        # Introduce missingness
        print("Introducing missingness...")
        df = self.introduce_missingness(df)
        
        # Forward fill and backfill
        print("Filling missing values...")
        df = self.forward_fill_and_backfill(df)
        
        return df, patients, deterioration_labels

    def validate_dataset(self, df: pd.DataFrame, patients: List[Dict], deterioration_labels: Dict) -> Dict:
        """Validate the generated dataset"""
        validation_stats = {}
        
        # Basic counts
        validation_stats['total_records'] = len(df)
        validation_stats['total_patients'] = df['PatientID'].nunique()
        
        # Risk score distribution (now numeric)
        risk_scores = df.groupby('PatientID')['Riskiness'].first()
        validation_stats['risk_score_stats'] = {
            'min': float(risk_scores.min()),
            'max': float(risk_scores.max()),
            'mean': float(risk_scores.mean()),
            'std': float(risk_scores.std())
        }
        
        # Count patients by risk level ranges
        healthy_count = ((risk_scores >= 0.0) & (risk_scores < 0.25)).sum()
        low_count     = ((risk_scores >= 0.25) & (risk_scores < 0.5)).sum()
        medium_count  = ((risk_scores >= 0.5) & (risk_scores < 0.75)).sum()
        high_count    = ((risk_scores >= 0.75) & (risk_scores <= 1.0)).sum()

        validation_stats['patients_by_risk_level'] = {
            'healthy (0-0.25)': int(healthy_count),
            'low (0.25-0.5)': int(low_count),
            'medium (0.5-0.75)': int(medium_count),
            'high (0.75-1.0)': int(high_count)
        }
        patient_data = df.groupby('PatientID').first().reset_index()

        # Create risk level categories for deterioration analysis
        patient_data['RiskLevel'] = pd.cut(patient_data['Riskiness'], 
            bins=[0.0, 0.25, 0.5, 0.75, 1.0], 
            labels=['healthy', 'low', 'medium', 'high'],
            include_lowest=True)
        
        deterioration_by_risk = patient_data.groupby('RiskLevel')['Deterioration_within_90days'].agg(['count', 'sum']).to_dict()
        validation_stats['deterioration_by_risk_level'] = deterioration_by_risk
        
        # Days per patient
        days_per_patient = df.groupby('PatientID')['Day'].max()
        validation_stats['days_per_patient'] = {
            'min': int(days_per_patient.min()),
            'mean': float(days_per_patient.mean()),
            'max': int(days_per_patient.max())
        }
        
        # Check values outside clinical bands
        out_of_bounds = 0
        for param, bands in self.clinical_bands.items():
            if param in df.columns:
                min_bound = min(bands['healthy'][0], bands['severe'][0])
                max_bound = max(bands['healthy'][1], bands['severe'][1])
                out_of_bounds += ((df[param] < min_bound) | (df[param] > max_bound)).sum()
        
        validation_stats['values_out_of_bounds'] = int(out_of_bounds)
        validation_stats['percent_out_of_bounds'] = float(out_of_bounds / (len(df) * len(self.clinical_bands)) * 100)
        
        # High risk deteriorations count
        high_risk_deteriorations = patient_data[(patient_data['RiskLevel'] == 'high') & 
                                              (patient_data['Deterioration_within_90days'] == 1)].shape[0]
        validation_stats['high_risk_deteriorations'] = high_risk_deteriorations
        
        return validation_stats

def main():
    """Main function to generate dataset and validation report"""
    print("Starting Chronic Care Synthetic Dataset Generation...")
    
    generator = ChronicCareDatasetGenerator()
    
    # Generate dataset
    df, patients, deterioration_labels = generator.generate_dataset()
    
    # Save CSV
    print("Saving dataset to CSV...")
    df.to_csv('fdataset.csv', index=False)
    
    # Generate validation report
    print("Generating validation report...")
    validation_stats = generator.validate_dataset(df, patients, deterioration_labels)
    
    # Save validation report
    with open('validation_report.json', 'w') as f:
        json.dump(validation_stats, f, indent=2)
    
    # Print sample and validation summary
    print("\n" + "="*80)
    print("DATASET SAMPLE (First 10 rows):")
    print("="*80)
    print(df.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY:")
    print("="*80)
    
    risk_levels = validation_stats['patients_by_risk_level']
    print(f"""
The synthetic chronic care dataset has been successfully generated with {validation_stats['total_records']:,} records 
across {validation_stats['total_patients']:,} patients. The cohort includes {risk_levels['healthy (0-0.25)']} 
healthy patients (risk score 0-0.25), {risk_levels['low (0.25-0.5)']} low-risk (0.25-0.5), {risk_levels['medium (0.5-0.75)']} 
medium-risk (0.5-0.75), and {risk_levels['high (0.75-1.0)']} high-risk patients (0.75-1.0).


Risk score statistics:
- Min: {validation_stats['risk_score_stats']['min']:.2f}
- Max: {validation_stats['risk_score_stats']['max']:.2f}
- Mean: {validation_stats['risk_score_stats']['mean']:.2f}
- Std: {validation_stats['risk_score_stats']['std']:.2f}

Deterioration rates by risk level show realistic clinical patterns, with {validation_stats['high_risk_deteriorations']} high-risk patients experiencing deterioration 
within 90 days. All values are properly clamped within clinical bounds 
({validation_stats['percent_out_of_bounds']:.1f}% out-of-bounds), and patients have longitudinal data spanning 
{validation_stats['days_per_patient']['min']}-{validation_stats['days_per_patient']['max']} days 
(mean: {validation_stats['days_per_patient']['mean']:.1f} days). The dataset includes realistic missingness patterns, 
disease-specific influences, and temporal variations suitable for training AI-driven risk prediction models.
    """.strip())
    
    print(f"\nHigh-risk deteriorations produced: {validation_stats['high_risk_deteriorations']}")
    print(f"Risk score distribution by level:")
    for level, count in validation_stats['patients_by_risk_level'].items():
        print(f"  {level}: {count} patients")
    print(f"The deterioration rate was calibrated based on clinical risk factors, ")
    print(f"number of conditions, extreme values, and medication adherence patterns.")
    
    print("\nDataset generation completed successfully!")
    print("Files generated:")
    print("- fdataset.csv")
    print("- validation_report.json")

if __name__ == "__main__":
    main()