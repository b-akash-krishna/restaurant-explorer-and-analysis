import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import optuna
import redis
import json
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Dataset.csv')

class RatingPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = []
        self.model_dir = os.path.join(BASE_DIR, 'models')
        self.model_file = os.path.join(self.model_dir, 'rating_predictor.pkl')
        self.encoders_file = os.path.join(self.model_dir, 'rating_encoders.pkl')
        self.features_file = os.path.join(self.model_dir, 'rating_features.pkl')
        self.feature_importance_file = os.path.join(self.model_dir, 'feature_importance.pkl')
        
        self.redis_client = self._connect_to_redis()
        self.feature_importance = {}
        self.evaluation_metrics = {}

    def _connect_to_redis(self):
        """Attempts to connect to Redis and returns the client object."""
        try:
            r = redis.StrictRedis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True,
            )
            r.ping()
            logger.info("Successfully connected to Redis.")
            return r
        except redis.ConnectionError as e:
            logger.error(f"Could not connect to Redis: {e}")
            return None

    def preprocess_data(self, df, fit_encoders=True):
        """Preprocess restaurant data for rating prediction"""
        df = df.copy()
        df = df.dropna(subset=['Aggregate rating'])

        categorical_cols = ['City', 'Cuisines', 'Rest type']

        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
                
                if fit_encoders or col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(df[col])
                    df[col] = self.encoders[col].transform(df[col])
                else:
                    unique_vals = df[col].unique()
                    unseen_labels_mask = ~np.isin(unique_vals, self.encoders[col].classes_)
                    unseen_labels = unique_vals[unseen_labels_mask]
                    
                    for label in unseen_labels:
                        self.encoders[col].classes_ = np.append(self.encoders[col].classes_, label)
                    
                    df[col] = self.encoders[col].transform(df[col])

        df = df.apply(pd.to_numeric, errors='coerce')
        
        feature_cols = ['Has Online delivery', 'Has Table booking', 'Votes', 'Cost', 'City', 'Cuisines', 'Rest type']
        self.feature_columns = [col for col in feature_cols if col in df.columns]

        X = df[self.feature_columns]
        y = df['Aggregate rating']

        X = X.dropna()
        y = y.loc[X.index]

        return X, y

    def train(self, data_path=DATA_PATH, n_estimators=100, max_depth=10, refit_encoders=True):
        """Trains the RandomForestRegressor model with comprehensive evaluation."""
        try:
            df = pd.read_csv(data_path)
            
            df.rename(columns={
                'Has Table booking': 'Has Table booking',
                'Has Online delivery': 'Has Online delivery',
                'Average Cost for two': 'Cost',
                'Restaurant Type': 'Rest type'
            }, inplace=True)
            
            if 'Has Table booking' in df.columns:
                df['Has Table booking'] = df['Has Table booking'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
            if 'Has Online delivery' in df.columns:
                df['Has Online delivery'] = df['Has Online delivery'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)

            X, y = self.preprocess_data(df, fit_encoders=refit_encoders)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

            # Predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate comprehensive metrics
            train_metrics = {
                'mse': mean_squared_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            }
            
            test_metrics = {
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
            
            self.evaluation_metrics = {
                'train': train_metrics,
                'test': test_metrics
            }
            
            # Feature importance analysis
            self.feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), 
                       key=lambda x: x[-1], 
                       reverse=True)
            )
            
            print(f"\n{'='*60}")
            print(f"Model Training Complete")
            print(f"{'='*60}")
            print(f"\nModel Configuration:")
            print(f"  n_estimators: {n_estimators}")
            print(f"  max_depth: {max_depth}")
            
            print(f"\nTraining Set Metrics:")
            for metric, value in train_metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            print(f"\nTest Set Metrics:")
            for metric, value in test_metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            print(f"\nFeature Importance (Top 5):")
            for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:5], 1):
                print(f"  {i}. {feature}: {importance:.4f}")
            
            print(f"\n{'='*60}\n")

            # Save everything
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.encoders, self.encoders_file)
            joblib.dump(self.feature_columns, self.features_file)
            joblib.dump(self.feature_importance, self.feature_importance_file)
            joblib.dump(self.evaluation_metrics, os.path.join(self.model_dir, 'evaluation_metrics.pkl'))

            return {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': self.feature_importance
            }

        except Exception as e:
            print(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def load_model(self):
        """Loads a pre-trained model and its encoders."""
        try:
            self.model = joblib.load(self.model_file)
            self.encoders = joblib.load(self.encoders_file)
            self.feature_columns = joblib.load(self.features_file)
            
            # Try to load feature importance if available
            try:
                self.feature_importance = joblib.load(self.feature_importance_file)
            except FileNotFoundError:
                self.feature_importance = {}
            
            # Try to load evaluation metrics if available
            try:
                self.evaluation_metrics = joblib.load(os.path.join(self.model_dir, 'evaluation_metrics.pkl'))
            except FileNotFoundError:
                self.evaluation_metrics = {}
            
            print("Pre-trained model, encoders, and features loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}. Please ensure you have run the training script first.")
            self.model = None

    def get_feature_importance(self):
        """Get feature importance data"""
        if not self.feature_importance and self.model:
            self.feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
        return self.feature_importance

    def get_model_interpretation(self):
        """Get comprehensive model interpretation data"""
        return {
            'feature_importance': self.get_feature_importance(),
            'evaluation_metrics': self.evaluation_metrics,
            'most_influential_features': list(self.feature_importance.keys())[:3] if self.feature_importance else []
        }

    def _create_cache_key(self, data: dict) -> str:
        """Generates a consistent cache key from input data."""
        sorted_items = sorted(data.items())
        return json.dumps(sorted_items)

    def predict_rating(self, data: dict):
        """Predicts the rating for a given restaurant using caching."""
        if self.redis_client:
            cache_key = self._create_cache_key(data)
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                logger.info("Cache hit for rating prediction.")
                return json.loads(cached_result)

        if not self.model:
            self.load_model()
            if not self.model:
                raise RuntimeError("Model is not loaded. Please train or load a model first.")

        df = pd.DataFrame([data])
        
        for col in ['City', 'Cuisines', 'Rest type']:
            if col in df.columns and col in self.encoders:
                le = self.encoders[col]
                df[col] = df[col].fillna('Unknown').astype(str)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                
                df[col] = le.transform(df[col])
            elif col not in df.columns:
                df[col] = 0

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        input_data = df[self.feature_columns]
        prediction = self.model.predict(input_data)[0]
        prediction = max(0.0, min(5.0, float(prediction)))

        if self.redis_client:
            self.redis_client.setex(cache_key, 3600, json.dumps(prediction))

        return prediction

    def objective(self, trial, data_path=DATA_PATH):
        """Objective function for Optuna hyperparameter tuning."""
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        
        try:
            df = pd.read_csv(data_path)
            
            df.rename(columns={
                'Has Table booking': 'Has Table booking',
                'Has Online delivery': 'Has Online delivery',
                'Average Cost for two': 'Cost',
                'Restaurant Type': 'Rest type'
            }, inplace=True)
            df['Has Table booking'] = df['Has Table booking'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
            df['Has Online delivery'] = df['Has Online delivery'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)

            X, y = self.preprocess_data(df, fit_encoders=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            return r2
            
        except FileNotFoundError:
            print(f"Data file not found at {data_path}")
            return -1.0
        except Exception as e:
            print(f"An error occurred during tuning trial: {e}")
            return -1.0
            
    def tune_hyperparameters(self, n_trials=50, data_path=DATA_PATH):
        """Performs hyperparameter tuning using Optuna and retrains the model with the best parameters."""
        print("Starting hyperparameter tuning...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, data_path), n_trials=n_trials)
        
        print("\nTuning finished.")
        print("Best trial:")
        print(f"  Value (R2 score): {study.best_value}")
        print("  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
            
        best_params = study.best_params
        print("Retraining model with best hyperparameters...")
        
        results = self.train(
            data_path=data_path,
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            refit_encoders=False
        )
        print("Model retrained and saved successfully!")
        return results

    def get_prediction_options(self):
        """Get unique values for dropdown options and a random sample"""
        try:
            df = pd.read_csv(DATA_PATH)
            
            rename_map = {}
            if 'Has Table booking' in df.columns:
                rename_map['Has Table booking'] = 'Has Table booking'
            if 'Has Online delivery' in df.columns:
                rename_map['Has Online delivery'] = 'Has Online delivery'
            if 'Average Cost for two' in df.columns:
                rename_map['Average Cost for two'] = 'Cost'
            if 'Restaurant Type' in df.columns:
                rename_map['Restaurant Type'] = 'Rest type'
            
            df.rename(columns=rename_map, inplace=True)
            
            if 'Has Table booking' in df.columns:
                df['Has Table booking'] = df['Has Table booking'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
            if 'Has Online delivery' in df.columns:
                df['Has Online delivery'] = df['Has Online delivery'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
            
            locations = []
            if 'City' in df.columns:
                locations = sorted(df['City'].dropna().unique().tolist())[:100]
            
            rest_types = []
            if 'Rest type' in df.columns:
                rest_types = sorted(df['Rest type'].dropna().unique().tolist())[:50]
            
            cuisines = []
            if 'Cuisines' in df.columns:
                for cuisine_str in df['Cuisines'].dropna():
                    cuisines.extend([c.strip() for c in str(cuisine_str).split(',')])
                cuisines = sorted(list(set(cuisines)))[:100]
            
            sample_row = df.sample(1).iloc[0]
            
            random_sample = {
                'votes': int(sample_row.get('Votes', 100)) if pd.notna(sample_row.get('Votes')) else 100,
                'online_order': int(sample_row.get('Has Online delivery', 1)) if pd.notna(sample_row.get('Has Online delivery')) else 1,
                'book_table': int(sample_row.get('Has Table booking', 0)) if pd.notna(sample_row.get('Has Table booking')) else 0,
                'location': str(sample_row.get('City', locations[0] if locations else 'Unknown')),
                'rest_type': str(sample_row.get('Rest type', rest_types[0] if rest_types else 'Casual Dining')),
                'cuisines': str(sample_row.get('Cuisines', cuisines[0] if cuisines else 'North Indian')),
                'cost': int(sample_row.get('Cost', 500)) if pd.notna(sample_row.get('Cost')) else 500
            }
            
            stats = {
                'votes_range': {
                    'min': int(df['Votes'].min()) if 'Votes' in df.columns else 0,
                    'max': int(df['Votes'].max()) if 'Votes' in df.columns else 10000,
                    'avg': int(df['Votes'].mean()) if 'Votes' in df.columns else 100
                },
                'cost_range': {
                    'min': int(df['Cost'].min()) if 'Cost' in df.columns else 0,
                    'max': int(df['Cost'].max()) if 'Cost' in df.columns else 5000,
                    'avg': int(df['Cost'].mean()) if 'Cost' in df.columns else 500
                }
            }
            
            if not locations:
                locations = ['Unknown']
            if not rest_types:
                rest_types = ['Casual Dining', 'Quick Bites', 'Cafe', 'Fine Dining']
            if not cuisines:
                cuisines = ['North Indian', 'Chinese', 'Continental', 'Italian']
            
            return {
                'locations': locations,
                'rest_types': rest_types,
                'cuisines': cuisines,
                'random_sample': random_sample,
                'stats': stats
            }
            
        except Exception as e:
            import traceback
            print(f"Error loading prediction options: {e}")
            traceback.print_exc()
            return {
                'locations': ['Bangalore', 'Delhi', 'Mumbai'],
                'rest_types': ['Casual Dining', 'Quick Bites', 'Cafe'],
                'cuisines': ['North Indian', 'Chinese', 'Continental'],
                'random_sample': {
                    'votes': 100,
                    'online_order': 1,
                    'book_table': 0,
                    'location': 'Bangalore',
                    'rest_type': 'Casual Dining',
                    'cuisines': 'North Indian',
                    'cost': 500
                },
                'stats': {
                    'votes_range': {'min': 0, 'max': 10000, 'avg': 100},
                    'cost_range': {'min': 0, 'max': 5000, 'avg': 500}
                }
            }


if __name__ == "__main__":
    predictor = RatingPredictor()
    
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at: {DATA_PATH}")
        exit(1)

    # Train with feature importance analysis
    results = predictor.train(data_path=DATA_PATH)
    print("\nTraining Results:")
    print(results)