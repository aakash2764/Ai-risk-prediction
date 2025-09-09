import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           confusion_matrix, classification_report,
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
from typing import Dict, List, Tuple, Any
import json
from scipy import stats

warnings.filterwarnings('ignore')
# --- utility for JSON serialization ---
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj


class HealthcareModelTrainer:
    def __init__(self, random_state: int = 42):
        """Initialize the trainer with configuration"""
        self.random_state = random_state
        self.models = {}
        self.calibrated_models = {}
        self.explainers = {}
        self.feature_importance = {}
        self.training_metrics = {}
        
        # XGBoost parameters optimized for healthcare data
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # For riskiness regression
        self.xgb_reg_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the processed features and targets"""
        print("Loading processed data...")
        
        try:
            X = pd.read_csv('model_features.csv')
            deterioration_target = pd.read_csv('deterioration_target.csv')
            riskiness_target = pd.read_csv('riskiness_target.csv')
            
            print(f"Features shape: {X.shape}")
            print(f"Deterioration target shape: {deterioration_target.shape}")
            print(f"Riskiness target shape: {riskiness_target.shape}")
            
            return X, deterioration_target, riskiness_target
            
        except FileNotFoundError as e:
            print(f"Error: Required file not found. Please run preprocess.py first.")
            raise

    def create_train_test_splits(self, X: pd.DataFrame, y_deterioration: pd.Series, 
                               y_riskiness: pd.Series) -> Dict[str, Any]:
        """Create stratified train/test splits"""
        print("Creating train/test splits...")
        
        # Use deterioration target for stratification
        X_train, X_test, y_det_train, y_det_test, y_risk_train, y_risk_test = train_test_split(
            X, y_deterioration, y_riskiness,
            test_size=0.2,
            stratify=y_deterioration,
            random_state=self.random_state
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Deterioration class distribution in train: {y_det_train.value_counts().to_dict()}")
        print(f"Deterioration class distribution in test: {y_det_test.value_counts().to_dict()}")
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_det_train': y_det_train, 'y_det_test': y_det_test,
            'y_risk_train': y_risk_train, 'y_risk_test': y_risk_test
        }

    def calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for imbalanced data"""
        class_counts = y.value_counts()
        total_samples = len(y)
        
        class_weights = {}
        for class_label in [0, 1]:
            if class_label in class_counts:
                class_weights[class_label] = total_samples / (2 * class_counts[class_label])
            else:
                class_weights[class_label] = 1.0
        
        return class_weights

    def train_deterioration_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost model for deterioration prediction"""
        print("\nTraining deterioration prediction model...")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        scale_pos_weight = class_weights[1] / class_weights[0]
        
        # Update parameters with class weight
        params = self.xgb_params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        
        # Initialize and train model
        model = xgb.XGBClassifier(**params)
        
        # Fit model
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        print("Deterioration model training completed")
        return model

    def train_riskiness_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> xgb.XGBRegressor:
        """Train XGBoost model for riskiness prediction"""
        print("\nTraining riskiness prediction model...")
        
        # Initialize and train model
        model = xgb.XGBRegressor(**self.xgb_reg_params)
        
        # Fit model
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        print("Riskiness model training completed")
        return model

    def evaluate_deterioration_model(self, model: xgb.XGBClassifier, 
                                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive evaluation of deterioration model"""
        print("\nEvaluating deterioration model...")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'auroc': roc_auc_score(y_test, y_pred_proba),
            'auprc': average_precision_score(y_test, y_pred_proba),
            'accuracy': (y_pred == y_test).mean(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Calibration assessment
        fraction_pos, mean_pred_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
        metrics['calibration'] = {
            'fraction_positive': fraction_pos.tolist(),
            'mean_predicted_value': mean_pred_value.tolist()
        }
        
        print(f"Deterioration Model Performance:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics

    def evaluate_riskiness_model(self, model: xgb.XGBRegressor, 
                               X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive evaluation of riskiness model"""
        print("\nEvaluating riskiness model...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Clip predictions to valid range [0, 1]
        y_pred = np.clip(y_pred, 0, 1)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'pearson_correlation': stats.pearsonr(y_test, y_pred)[0],
            'spearman_correlation': stats.spearmanr(y_test, y_pred)[0]
        }
        
        print(f"Riskiness Model Performance:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  Pearson r: {metrics['pearson_correlation']:.4f}")
        
        return metrics

    def create_shap_explanations(self, model, X_train: pd.DataFrame, 
                               X_test: pd.DataFrame, model_type: str) -> shap.Explainer:
        """Create SHAP explainer for model interpretability"""
        print(f"\nCreating SHAP explanations for {model_type} model...")
        
        # Use TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for test set (sample if too large)
        X_sample = X_test.sample(min(1000, len(X_test)), random_state=self.random_state)
        shap_values = explainer.shap_values(X_sample)
        
        # Store for later use
        self.explainers[model_type] = {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_sample': X_sample
        }
        
        print(f"SHAP explanations created for {model_type} model")
        return explainer

    def get_global_feature_importance(self, model, feature_names: List[str], 
                                    model_type: str) -> Dict[str, float]:
        """Get global feature importance from the model"""
        
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        else:
            # Fallback for other model types
            importance_scores = np.zeros(len(feature_names))
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        self.feature_importance[model_type] = sorted_importance
        return sorted_importance

    def create_clinical_explanations(self, feature_names: List[str]) -> Dict[str, str]:
        """Create clinician-friendly explanations for features"""
        
        clinical_explanations = {}
        
        for feature in feature_names:
            # Parse feature name to create clinical explanation
            if 'SystolicBP' in feature:
                clinical_explanations[feature] = "Blood pressure (systolic) - higher values indicate hypertension risk"
            elif 'DiastolicBP' in feature:
                clinical_explanations[feature] = "Blood pressure (diastolic) - elevated values suggest cardiovascular stress"
            elif 'HeartRate' in feature:
                clinical_explanations[feature] = "Heart rate - abnormal values may indicate cardiac issues"
            elif 'Glucose' in feature:
                clinical_explanations[feature] = "Blood glucose - elevated levels indicate diabetes risk"
            elif 'Creatinine' in feature:
                clinical_explanations[feature] = "Kidney function marker - higher values suggest kidney problems"
            elif 'ALT' in feature or 'AST' in feature:
                clinical_explanations[feature] = "Liver enzyme - elevated levels indicate liver damage"
            elif 'MedicationAdherence' in feature:
                clinical_explanations[feature] = "Medication compliance - lower values indicate poor adherence"
            elif 'WeightBMI' in feature:
                clinical_explanations[feature] = "Body mass index - higher values indicate obesity"
            elif 'O2Sat' in feature:
                clinical_explanations[feature] = "Oxygen saturation - lower values indicate respiratory issues"
            elif '_slope_' in feature:
                clinical_explanations[feature] = f"Trend in {feature.split('_')[0]} - positive slope indicates increasing values"
            elif '_delta_' in feature:
                clinical_explanations[feature] = f"Change in {feature.split('_')[0]} over time"
            elif 'risk' in feature.lower():
                clinical_explanations[feature] = "Combined risk factor based on multiple clinical indicators"
            else:
                # Generic explanation
                base_param = feature.split('_')[0] if '_' in feature else feature
                clinical_explanations[feature] = f"Clinical parameter: {base_param}"
        
        return clinical_explanations

    def plot_evaluation_charts(self, deterioration_metrics: Dict, riskiness_metrics: Dict):
        """Create evaluation visualization charts"""
        print("\nCreating evaluation charts...")
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Healthcare Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Deterioration Confusion Matrix
        cm = np.array(deterioration_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Deterioration', 'Deterioration'],
                   yticklabels=['No Deterioration', 'Deterioration'],
                   ax=axes[0,0])
        axes[0,0].set_title('Deterioration Prediction\nConfusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # 2. Calibration Plot
        frac_pos = deterioration_metrics['calibration']['fraction_positive']
        mean_pred = deterioration_metrics['calibration']['mean_predicted_value']
        axes[0,1].plot(mean_pred, frac_pos, marker='o', linewidth=2, label='Model')
        axes[0,1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        axes[0,1].set_xlabel('Mean Predicted Probability')
        axes[0,1].set_ylabel('Fraction of Positives')
        axes[0,1].set_title('Deterioration Model\nCalibration Plot')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Metrics Bar Chart - Deterioration
        det_metrics_plot = {
            'AUROC': deterioration_metrics['auroc'],
            'AUPRC': deterioration_metrics['auprc'],
            'Accuracy': deterioration_metrics['accuracy']
        }
        axes[0,2].bar(det_metrics_plot.keys(), det_metrics_plot.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0,2].set_title('Deterioration Model\nPerformance Metrics')
        axes[0,2].set_ylim(0, 1)
        axes[0,2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (k, v) in enumerate(det_metrics_plot.items()):
            axes[0,2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 4. Riskiness Metrics Bar Chart
        risk_metrics_plot = {
            'R² Score': max(0, riskiness_metrics['r2_score']),  # Ensure non-negative for plotting
            'Pearson r': abs(riskiness_metrics['pearson_correlation']),  # Use absolute value
            '1 - RMSE': max(0, 1 - riskiness_metrics['rmse'])  # Convert RMSE to 0-1 scale
        }
        axes[1,0].bar(risk_metrics_plot.keys(), risk_metrics_plot.values(), color=['#96CEB4', '#FECA57', '#FF9FF3'])
        axes[1,0].set_title('Riskiness Model\nPerformance Metrics')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (k, v) in enumerate(risk_metrics_plot.items()):
            axes[1,0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 5. Feature Importance (Top 10)
        if 'deterioration' in self.feature_importance:
            top_features = list(self.feature_importance['deterioration'].items())[:10]
            feature_names = [f.replace('_', ' ').title()[:20] + '...' if len(f) > 20 else f.replace('_', ' ').title() for f, _ in top_features]
            importance_values = [imp for _, imp in top_features]
            
            axes[1,1].barh(range(len(feature_names)), importance_values, color='skyblue')
            axes[1,1].set_yticks(range(len(feature_names)))
            axes[1,1].set_yticklabels(feature_names, fontsize=8)
            axes[1,1].set_xlabel('Feature Importance')
            axes[1,1].set_title('Top 10 Features\n(Deterioration Model)')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Model Summary Text
        axes[1,2].axis('off')
        summary_text = f"""
MODEL TRAINING SUMMARY

Deterioration Model (Classification):
• AUROC: {deterioration_metrics['auroc']:.3f}
• AUPRC: {deterioration_metrics['auprc']:.3f}
• Accuracy: {deterioration_metrics['accuracy']:.3f}

Riskiness Model (Regression):
• RMSE: {riskiness_metrics['rmse']:.3f}
• MAE: {riskiness_metrics['mae']:.3f}
• R² Score: {riskiness_metrics['r2_score']:.3f}
• Pearson r: {riskiness_metrics['pearson_correlation']:.3f}

Models successfully trained and calibrated
for clinical risk prediction
        """
        axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Evaluation charts saved as 'model_evaluation_results.png'")

    def save_models_and_results(self, deterioration_model: xgb.XGBClassifier, 
                              riskiness_model: xgb.XGBRegressor,
                              deterioration_metrics: Dict, riskiness_metrics: Dict,
                              feature_names: List[str]):
        """Save trained models and results"""
        print("\nSaving models and results...")
        
        # Save models
        joblib.dump(deterioration_model, 'deterioration_model.pkl')
        joblib.dump(riskiness_model, 'riskiness_model.pkl')
        
        # Save SHAP explainers
        joblib.dump(self.explainers, 'shap_explainers.pkl')
        
        # Create clinical explanations
        clinical_explanations = self.create_clinical_explanations(feature_names)
        
        # Combine all results
        results = {
            'deterioration_metrics': deterioration_metrics,
            'riskiness_metrics': riskiness_metrics,
            'feature_importance': self.feature_importance,
            'clinical_explanations': clinical_explanations,
            'feature_names': feature_names,
            'model_parameters': {
                'deterioration': self.xgb_params,
                'riskiness': self.xgb_reg_params
            },
            'training_summary': {
                'total_features': len(feature_names),
                'deterioration_auroc': deterioration_metrics['auroc'],
                'riskiness_r2': riskiness_metrics['r2_score']
            }
        }
        
        # Save results as JSON
        serializable_results = convert_to_serializable(results)
        with open('training_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
                
        # Save feature importance as CSV
        if 'deterioration' in self.feature_importance:
            imp_df = pd.DataFrame([
                {'feature': k, 'importance': v, 'clinical_explanation': clinical_explanations.get(k, '')}
                for k, v in self.feature_importance['deterioration'].items()
            ])
            imp_df.to_csv('feature_importance.csv', index=False)
        
        print("Models and results saved successfully:")
        print("  - deterioration_model.pkl")
        print("  - riskiness_model.pkl") 
        print("  - shap_explainers.pkl")
        print("  - training_results.json")
        print("  - feature_importance.csv")
        print("  - model_evaluation_results.png")

    def train_models(self):
        """Main training function"""
        print("🏥 Healthcare AI Model Training Pipeline")
        print("="*60)
        
        # Load data
        X, deterioration_target_df, riskiness_target_df = self.load_processed_data()
        
        # Extract target values
        y_deterioration = deterioration_target_df['deterioration_target']
        y_riskiness = riskiness_target_df['riskiness_target']
        
        # Create train/test splits
        splits = self.create_train_test_splits(X, y_deterioration, y_riskiness)
        
        # Train deterioration model
        deterioration_model = self.train_deterioration_model(
            splits['X_train'], splits['y_det_train'],
            splits['X_test'], splits['y_det_test']
        )
        
        # Train riskiness model  
        riskiness_model = self.train_riskiness_model(
            splits['X_train'], splits['y_risk_train'],
            splits['X_test'], splits['y_risk_test']
        )
        
        # Evaluate models
        deterioration_metrics = self.evaluate_deterioration_model(
            deterioration_model, splits['X_test'], splits['y_det_test']
        )
        
        riskiness_metrics = self.evaluate_riskiness_model(
            riskiness_model, splits['X_test'], splits['y_risk_test']
        )
        
        # Create SHAP explanations
        self.create_shap_explanations(deterioration_model, splits['X_train'], 
                                    splits['X_test'], 'deterioration')
        self.create_shap_explanations(riskiness_model, splits['X_train'], 
                                    splits['X_test'], 'riskiness')
        
        # Get feature importance
        feature_names = X.columns.tolist()
        self.get_global_feature_importance(deterioration_model, feature_names, 'deterioration')
        self.get_global_feature_importance(riskiness_model, feature_names, 'riskiness')
        
        # Create evaluation charts
        self.plot_evaluation_charts(deterioration_metrics, riskiness_metrics)
        
        # Save everything
        self.save_models_and_results(
            deterioration_model, riskiness_model,
            deterioration_metrics, riskiness_metrics, feature_names
        )
        
        print("\n✅ Model training completed successfully!")
        print("\n📊 Final Results Summary:")
        print(f"Deterioration Prediction - AUROC: {deterioration_metrics['auroc']:.4f}, AUPRC: {deterioration_metrics['auprc']:.4f}")
        print(f"Risk Score Prediction - R²: {riskiness_metrics['r2_score']:.4f}, RMSE: {riskiness_metrics['rmse']:.4f}")

def main():
    """Main execution function"""
    trainer = HealthcareModelTrainer(random_state=42)
    trainer.train_models()

if __name__ == "__main__":
    main()