import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class HealthcareFeatureEngineer:
    def __init__(self):
        """Initialize feature engineering configuration"""
        # Feature groups with their respective windows and aggregations
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
        
        # Outlier boundaries for key features (helps with robustness)
        self.outlier_bounds = {
            'SystolicBP': (50, 250),
            'DiastolicBP': (30, 150),
            'HeartRate': (30, 180),
            'RespirationRate': (5, 40),
            'O2Sat': (70, 100),
            'WeightBMI': (10, 60),
            'Glucose': (40, 500),
            'Cholesterol': (50, 400),
            'Creatinine': (0.1, 15),
            'Sodium': (120, 160),
            'Potassium': (2.0, 7.0),
            'Albumin': (1.0, 6.0),
            'AST': (5, 500),
            'ALT': (5, 500),
            'Bilirubin': (0.1, 25),
            'SleepHours': (2, 16),
            'ExerciseMinutes': (0, 300),
            'AlcoholIntake': (0, 20),
            'SmokingCigs': (0, 60)
        }
        
        self.feature_counts = {
            'vitals': 0,
            'labs': 0, 
            'medication': 0,
            'lifestyle': 0,
            'derived': 0,
            'temporal': 0,
            'interaction': 0
        }

    def load_and_validate_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate the dataset"""
        print(f"Loading dataset from {filepath}...")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} records for {df['PatientID'].nunique():,} patients")
        
        # Required columns - updated for multi-target
        required_cols = ['PatientID', 'Day', 'Deterioration_within_90days', 'Riskiness'] + \
                       self.feature_config['vitals']['features'] + \
                       self.feature_config['labs']['features'] + \
                       self.feature_config['medication']['features'] + \
                       self.feature_config['lifestyle']['features']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert Day to datetime if it's not already
        if df['Day'].dtype == 'object':
            df['Day'] = pd.to_datetime(df['Day'])
        
        # Sort by PatientID and Day
        df = df.sort_values(['PatientID', 'Day']).reset_index(drop=True)
        
        # Apply outlier clipping to improve robustness
        df = self.apply_outlier_clipping(df)
        
        print("Dataset validation and outlier clipping completed successfully")
        return df

    def apply_outlier_clipping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier clipping to improve model robustness"""
        print("Applying outlier clipping for robustness...")
        
        for feature, (lower, upper) in self.outlier_bounds.items():
            if feature in df.columns:
                original_outliers = ((df[feature] < lower) | (df[feature] > upper)).sum()
                df[feature] = np.clip(df[feature], lower, upper)
                if original_outliers > 0:
                    print(f"  Clipped {original_outliers} outliers in {feature}")
        
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
            # No data available, return zeros/NaNs
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

    def create_temporal_delta_features(self, patient_features: Dict[str, float]) -> Dict[str, float]:
        """Create temporal delta/change features for better trend detection"""
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

        self.feature_counts['temporal'] = len(temporal)
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
        alcohol_risk = min(alcohol_30d / 10, 1.0)  # >10 drinks high risk
        smoking_risk = min(smoking_30d / 20, 1.0)  # >20 cigs high risk
        sleep_risk = abs(sleep_30d - 7.5) / 7.5  # Deviation from optimal 7.5h
        
        interactions['Lifestyle_risk_score'] = (alcohol_risk + smoking_risk + sleep_risk) / 3
        
        self.feature_counts['interaction'] = len(interactions)
        return interactions

    def create_derived_features(self, patient_features: Dict[str, float]) -> Dict[str, float]:
        """Create derived clinical features"""
        derived = {}
        
        # Pulse Pressure features (SystolicBP - DiastolicBP)
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

        # Combined lifestyle features (weighted averages instead of many separate features)
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

        self.feature_counts['derived'] = len(derived)
        return derived

    def process_patient_group(self, patient_data: pd.DataFrame) -> Dict[str, float]:
        """Process a single patient's data to create features"""
        
        # Get the last recorded day as cutoff
        cutoff_day = patient_data['Day'].max()
        
        # Get both target variables from the cutoff day
        cutoff_row = patient_data[patient_data['Day'] == cutoff_day].iloc[0]
        deterioration_target = int(cutoff_row['Deterioration_within_90days']) if not pd.isna(cutoff_row['Deterioration_within_90days']) else 0
        riskiness_target = float(cutoff_row['Riskiness']) if not pd.isna(cutoff_row['Riskiness']) else 0.0
        
        patient_features = {
            'PatientID': patient_data['PatientID'].iloc[0], 
            'deterioration_target': deterioration_target,
            'riskiness_target': riskiness_target
        }
        
        # Process each feature group
        for group_name, config in self.feature_config.items():
            group_feature_count = 0
            
            for feature in config['features']:
                for window in config['windows']:
                    feature_results = self.aggregate_features(
                        patient_data, feature, window, cutoff_day, config['aggregations']
                    )
                    patient_features.update(feature_results)
                    group_feature_count += len(feature_results)
            
            self.feature_counts[group_name] = group_feature_count

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

    def calculate_class_weights(self, feature_df: pd.DataFrame) -> Dict[int, float]:
        """Calculate class weights for handling imbalanced dataset (for deterioration target)"""
        target_counts = feature_df['deterioration_target'].value_counts()
        total_samples = len(feature_df)
        
        class_weights = {}
        for class_label in [0, 1]:
            if class_label in target_counts:
                class_weights[class_label] = total_samples / (2 * target_counts[class_label])
            else:
                class_weights[class_label] = 1.0
        
        return class_weights

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering function"""
        print("Starting feature engineering...")
        
        # Group by patient and process
        patient_features = []
        
        total_patients = df['PatientID'].nunique()
        processed = 0
        
        for patient_id, patient_data in df.groupby('PatientID'):
            patient_results = self.process_patient_group(patient_data)
            patient_features.append(patient_results)
            
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed:,}/{total_patients:,} patients ({processed/total_patients*100:.1f}%)")
        
        print(f"Feature engineering completed for {processed:,} patients")
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(patient_features)
        
        # Ensure all numeric features (drop object types)
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'PatientID' not in numeric_cols:
            numeric_cols = ['PatientID'] + numeric_cols
        if 'deterioration_target' not in numeric_cols:
            numeric_cols.append('deterioration_target')
        if 'riskiness_target' not in numeric_cols:
            numeric_cols.append('riskiness_target')
            
        feature_df = feature_df[numeric_cols]
        
        # Fill any remaining NaN values with 0
        feature_df = feature_df.fillna(0)
        
        # Calculate and save class weights for deterioration prediction
        class_weights = self.calculate_class_weights(feature_df)
        print(f"Deterioration class weights calculated: {class_weights}")
        
        return feature_df

    def save_outputs(self, feature_df: pd.DataFrame) -> None:
        """Save processed features to CSV files"""
        print("Saving output files...")
        
        # Calculate class weights for deterioration prediction
        class_weights = self.calculate_class_weights(feature_df)
        
        # Save full feature matrix
        feature_df.to_csv('processed_features.csv', index=False)
        print("✓ Saved processed_features.csv")
        
        # Save model features (X) - exclude PatientID and both targets
        model_features = feature_df.drop(['PatientID', 'deterioration_target', 'riskiness_target'], axis=1)
        model_features.to_csv('model_features.csv', index=False)
        print("✓ Saved model_features.csv")
        
        # Save both target variables
        deterioration_target_df = feature_df[['PatientID', 'deterioration_target']]
        deterioration_target_df.to_csv('deterioration_target.csv', index=False)
        print("✓ Saved deterioration_target.csv")
        
        riskiness_target_df = feature_df[['PatientID', 'riskiness_target']]
        riskiness_target_df.to_csv('riskiness_target.csv', index=False)
        print("✓ Saved riskiness_target.csv")
        
        # Save combined targets for multi-task learning
        combined_targets_df = feature_df[['PatientID', 'deterioration_target', 'riskiness_target']]
        combined_targets_df.to_csv('combined_targets.csv', index=False)
        print("✓ Saved combined_targets.csv")
        
        # Save class weights for deterioration prediction
        weights_df = pd.DataFrame([class_weights]).T.reset_index()
        weights_df.columns = ['class', 'weight']
        weights_df.to_csv('deterioration_class_weights.csv', index=False)
        print("✓ Saved deterioration_class_weights.csv")

    def print_summary(self, feature_df: pd.DataFrame) -> None:
        """Print dataset summary and feature breakdown"""
        print("\n" + "="*80)
        print("ENHANCED MULTI-TARGET FEATURE ENGINEERING SUMMARY")
        print("="*80)
        
        # Dataset shape
        print(f"Final dataset shape: {feature_df.shape}")
        print(f"Number of patients: {len(feature_df):,}")
        print(f"Number of features: {feature_df.shape[1] - 3:,} (excluding PatientID and both targets)")  # -3 for PatientID and both targets
        
        # Deterioration target distribution
        deterioration_dist = feature_df['deterioration_target'].value_counts().sort_index()
        print(f"\nDeterioration target distribution:")
        print(f"  No deterioration (0): {deterioration_dist.get(0, 0):,} patients ({deterioration_dist.get(0, 0)/len(feature_df)*100:.1f}%)")
        print(f"  Deterioration (1):    {deterioration_dist.get(1, 0):,} patients ({deterioration_dist.get(1, 0)/len(feature_df)*100:.1f}%)")
        
        # Riskiness target statistics
        riskiness_stats = feature_df['riskiness_target'].describe()
        print(f"\nRiskiness target statistics:")
        print(f"  Min: {riskiness_stats['min']:.3f}")
        print(f"  Max: {riskiness_stats['max']:.3f}")
        print(f"  Mean: {riskiness_stats['mean']:.3f}")
        print(f"  Std: {riskiness_stats['std']:.3f}")
        
        # Risk level distribution (binned)
        risk_bins = pd.cut(feature_df['riskiness_target'], 
                          bins=[0.0, 0.25, 0.5, 0.75, 1.0], 
                          labels=['Low (0-0.25)', 'Medium-Low (0.25-0.5)', 'Medium-High (0.5-0.75)', 'High (0.75-1.0)'],
                          include_lowest=True)
        risk_dist = risk_bins.value_counts().sort_index()
        print(f"\nRiskiness distribution by level:")
        for level, count in risk_dist.items():
            print(f"  {level}: {count:,} patients ({count/len(feature_df)*100:.1f}%)")
        
        # Class weights
        class_weights = self.calculate_class_weights(feature_df)
        print(f"\nDeterioration class weights: {class_weights}")
        
        # Feature group breakdown
        print(f"\nFeature breakdown by group:")
        total_features = sum(self.feature_counts.values())
        for group, count in self.feature_counts.items():
            if count > 0:
                print(f"  {group.capitalize()}: {count:,} features ({count/total_features*100:.1f}%)")
        
        print(f"  Total: {total_features:,} features")
        
        # Sample feature names by group
        feature_cols = [col for col in feature_df.columns if col not in ['PatientID', 'deterioration_target', 'riskiness_target']]
        
        print(f"\nSample enhanced features:")
        temporal_features = [f for f in feature_cols if 'delta' in f or 'pct_change' in f][:3]
        if temporal_features:
            print(f"  Temporal: {', '.join(temporal_features)}")
        
        interaction_features = [f for f in feature_cols if any(x in f for x in ['risk', 'ratio', 'combined'])][:3]
        if interaction_features:
            print(f"  Interactions: {', '.join(interaction_features)}")
        
        ema_features = [f for f in feature_cols if 'ema' in f][:3]
        if ema_features:
            print(f"  EMA trends: {', '.join(ema_features)}")

def main():
    """Main preprocessing function"""
    print("🏥 Enhanced Healthcare Dataset Multi-Target Feature Engineering")
    print("="*80)
    
    # Initialize feature engineer
    engineer = HealthcareFeatureEngineer()
    
    try:
        # Load and validate data
        df = engineer.load_and_validate_data('fdataset.csv')
        
        # Engineer features
        feature_df = engineer.engineer_features(df)
        
        # Save outputs
        engineer.save_outputs(feature_df)
        
        # Print summary
        engineer.print_summary(feature_df)
        
        print(f"\n✅ Enhanced multi-target preprocessing completed successfully!")
        print("Generated files:")
        print("  - processed_features.csv (full feature matrix)")
        print("  - model_features.csv (X - features only)")
        print("  - deterioration_target.csv (y1 - deterioration target)")
        print("  - riskiness_target.csv (y2 - riskiness target)")
        print("  - combined_targets.csv (both targets for multi-task learning)")
        print("  - deterioration_class_weights.csv (for handling deterioration class imbalance)")
        print("\n🎯 Key improvements for multi-target prediction:")
        print("  ✓ Dual target variables: deterioration (classification) + riskiness (regression)")
        print("  ✓ Outlier clipping for robustness")
        print("  ✓ Temporal delta/trend features")
        print("  ✓ Clinical interaction features")
        print("  ✓ EMA smoothing for noise reduction")
        print("  ✓ Combined lifestyle risk scores")
        print("  ✓ Class weights calculated for deterioration imbalance")
        print("  ✓ Separate and combined target files for flexible modeling approaches")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()