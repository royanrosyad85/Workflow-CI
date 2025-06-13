import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import warnings
import os
import argparse
import sys
from dotenv import load_dotenv

# Fix matplotlib backend
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

class WaterPotabilityCICD:
    
    def __init__(self, experiment_name="Water_Potability_CI"):
        self.experiment_name = experiment_name
        self.use_remote_tracking = False
        self.setup_environment()
        self.setup_mlflow()
        
    def setup_environment(self):
        # Load environment variables from .env file
        load_dotenv()
        
        print("🔧 Setting up environment configuration...")
        
        # Check if required environment variables are present
        dagshub_token = os.environ.get('DAGSHUB_TOKEN')
        dagshub_username = os.environ.get('DAGSHUB_USERNAME')
        
        if dagshub_token and dagshub_username:
            print("✅ DagsHub credentials found in environment")
            self.use_remote_tracking = True
        else:
            print("⚠️ DagsHub credentials not found. Will use local tracking")
            self.use_remote_tracking = False
        
    def setup_mlflow(self):
        # Setup MLflow tracking based on available credentials
        if self.use_remote_tracking:
            # Check if DagsHub credentials are available
            if os.environ.get('DAGSHUB_TOKEN') and os.environ.get('DAGSHUB_USERNAME'):
                # Set up MLflow tracking with DagsHub
                os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USERNAME')
                os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN')
                
                # Set tracking URI menggunakan environment variable
                dagshub_username = os.environ.get('DAGSHUB_USERNAME')
                tracking_uri = f"https://dagshub.com/{dagshub_username}/Water-Quality-Tracking-Model.mlflow"
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                
                print("✅ Using remote MLflow tracking on DagsHub")
                print(f"📊 Tracking URI: {tracking_uri}")
                print(f"👤 Username: {dagshub_username}")
            else:
                # Fallback to local if credentials not properly set
                self.use_remote_tracking = False
                self._setup_local_tracking()
        else:
            self._setup_local_tracking()
    
    def _setup_local_tracking(self):
        # Use local tracking if credentials aren't available
        print("🏠 DagsHub credentials not found. Using local MLflow tracking")
        os.makedirs('mlruns', exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("water_potability_local")
        print("📁 Local MLflow tracking initialized")
        
    def create_sample_data(self):
        # Create sample data if file doesn't exist
        print("📂 Creating sample data...")
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'ph': np.random.normal(7.0, 1.5, n_samples),
            'Hardness': np.random.normal(200, 30, n_samples),
            'Solids': np.random.normal(22000, 8000, n_samples),
            'Chloramines': np.random.normal(7.0, 1.5, n_samples),
            'Sulfate': np.random.normal(330, 40, n_samples),
            'Conductivity': np.random.normal(420, 80, n_samples),
            'Organic_carbon': np.random.normal(14, 3, n_samples),
            'Trihalomethanes': np.random.normal(66, 16, n_samples),
            'Turbidity': np.random.normal(4.0, 0.8, n_samples),
            'Potability': np.random.choice([0, 1], n_samples, p=[0.61, 0.39])
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        df.loc[missing_indices[:50], 'ph'] = np.nan
        df.loc[missing_indices[50:], 'Sulfate'] = np.nan
        
        return df
    
    def load_and_preprocess_data(self, data_path):
        # Load or create data
        if os.path.exists(data_path):
            print(f"📂 Loading dataset from {data_path}...")
            df = pd.read_csv(data_path)
        else:
            print(f"⚠️ File {data_path} not found. Creating sample data...")
            df = self.create_sample_data()
        
        # Handle missing values
        df.fillna(df.median(), inplace=True)
        
        # Separate features and target
        X = df.drop('Potability', axis=1)
        y = df['Potability']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        oversampler = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
        
        # Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to numpy arrays
        X_train_scaled = np.array(X_train_scaled)
        X_test_scaled = np.array(X_test_scaled)
        y_train_resampled = np.array(y_train_resampled)
        y_test = np.array(y_test)
        
        print(f"📊 Data processed: {len(X_train_scaled)} train, {len(X_test_scaled)} test samples")
        
        return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler
    
    def train_model(self, model, model_name, X_train, X_test, y_train, y_test):
        # Train model with MLflow tracking
        run_name = f"{model_name}_CI_{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"🚀 Training {model_name} in CI/CD pipeline...")
            
            # Enable autologging
            mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=True)
            
            # Log CI/CD specific parameters
            mlflow.log_param("pipeline_type", "CI/CD")
            mlflow.log_param("environment", "docker" if os.environ.get('DOCKER_ENV') else "local")
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("tracking_mode", "remote" if self.use_remote_tracking else "local")
            
            # Log GitHub Actions info if available
            if os.environ.get('GITHUB_SHA'):
                mlflow.log_param("github_sha", os.environ.get('GITHUB_SHA'))
                mlflow.log_param("github_ref", os.environ.get('GITHUB_REF'))
                mlflow.log_param("github_run_number", os.environ.get('GITHUB_RUN_NUMBER'))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Log additional metrics
            mlflow.log_metric("ci_accuracy", accuracy)
            mlflow.log_metric("ci_f1_score", f1)
            mlflow.log_metric("ci_precision", precision)
            mlflow.log_metric("ci_recall", recall)
            
            # Create model output directory
            os.makedirs("model_output", exist_ok=True)
            
            # Save model locally for artifact upload
            model_path = f"model_output/{model_name}_model.pkl"
            mlflow.sklearn.save_model(model, model_path)
            
            print(f"✅ {model_name} CI training completed!")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   📊 F1 Score: {f1:.4f}")
            print(f"   🔄 Tracking: {'Remote (DagsHub)' if self.use_remote_tracking else 'Local'}")
            
            return model, accuracy, f1
    
    def run_ci_pipeline(self, data_path):
        # Run complete CI/CD pipeline
        print("🔄 Starting CI/CD Pipeline...")
        print(f"🌐 Environment: {'Remote (DagsHub)' if self.use_remote_tracking else 'Local'}")
        
        # Load data
        X_train, X_test, y_train, y_test, scaler = self.load_and_preprocess_data(data_path)
        
        # Define models for CI
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=500),
            "SVM": SVC(random_state=42, probability=True, kernel='rbf')
        }
        
        # Train models and collect results
        results = {}
        best_model = None
        best_score = 0
        
        for model_name, model in models.items():
            try:
                trained_model, accuracy, f1 = self.train_model(
                    model, model_name, X_train, X_test, y_train, y_test
                )
                results[model_name] = {
                    'model': trained_model,
                    'accuracy': accuracy,
                    'f1_score': f1
                }
                
                # Track best model
                if f1 > best_score:
                    best_score = f1
                    best_model = model_name
                    
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
        
        # Log pipeline summary
        summary_run_name = f"CI_Pipeline_Summary_{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"
        with mlflow.start_run(run_name=summary_run_name):
            mlflow.log_param("total_models_trained", len(results))
            mlflow.log_param("best_model", best_model)
            mlflow.log_param("tracking_mode", "remote" if self.use_remote_tracking else "local")
            mlflow.log_metric("best_f1_score", best_score)
            
            # Log environment info
            mlflow.log_param("dagshub_username", os.environ.get('DAGSHUB_USERNAME', 'not_set'))
            mlflow.log_param("docker_username", os.environ.get('DOCKER_HUB_USERNAME', 'not_set'))
            
            # Log all model performances
            for name, result in results.items():
                mlflow.log_metric(f"{name}_accuracy", result['accuracy'])
                mlflow.log_metric(f"{name}_f1_score", result['f1_score'])
        
        print(f"\n🎉 CI/CD Pipeline completed!")
        print(f"📊 Best model: {best_model} (F1: {best_score:.4f})")
        print(f"📁 Model artifacts saved to: model_output/")
        print(f"🔄 Tracking mode: {'Remote (DagsHub)' if self.use_remote_tracking else 'Local'}")
        
        return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Water Potability CI/CD Pipeline')
    parser.add_argument('--data_path', type=str, default='water_potability_preprocessed.csv',
                       help='Path to the dataset')
    parser.add_argument('--experiment_name', type=str, default='Water_Potability_CI',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    try:
        # Initialize CI/CD pipeline
        pipeline = WaterPotabilityCICD(experiment_name=args.experiment_name)
        
        # Run pipeline
        results = pipeline.run_ci_pipeline(args.data_path)
        
        print("\n🎯 CI/CD Pipeline execution successful!")
        
    except Exception as e:
        print(f"❌ CI/CD Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()