import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import optuna
import redis
import json
import logging

logger = logging.getLogger(__name__)


class RatingPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = []
        self.redis_client = self._connect_to_redis()

    def _connect_to_redis(self):
        """Attempts to connect to Redis and returns the client object."""
        try:
            r = redis.StrictRedis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True,  # Decode responses to get strings instead of bytes
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

        # Drop rows with missing target variable
        df = df.dropna(subset=['Aggregate rating'])

        categorical_cols = ['City', 'Cuisines', 'Has Table booking', 'Has Online delivery', 'Price range']

        for col in categorical_cols:
            if col in df.columns:
                # Fill missing values first
                df[col] = df[col].fillna('Unknown').astype(str)
                
                if fit_encoders or col not in self.encoders:
                    # Fit new encoder
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(df[col])
                    df[col] = self.encoders[col].transform(df[col])
                else:
                    # Transform using existing encoder, handle unseen labels
                    # Get unique values in current data
                    unique_vals = df[col].unique()
                    
                    # Create a mapping for unseen values to a default (e.g., -1)
                    # NOTE: This approach is simpler but can be improved with a more robust strategy
                    # Here we transform existing labels and set new ones to -1
                    unseen_labels_mask = ~np.isin(unique_vals, self.encoders[col].classes_)
                    unseen_labels = unique_vals[unseen_labels_mask]
                    
                    for label in unseen_labels:
                        self.encoders[col].classes_ = np.append(self.encoders[col].classes_, label)
                    
                    df[col] = self.encoders[col].transform(df[col])


        # Ensure all columns are numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Define features and target
        feature_cols = ['Has Online delivery', 'Has Table booking', 'Votes', 'Cost', 'City', 'Cuisines', 'Rest type']
        self.feature_columns = [col for col in feature_cols if col in df.columns]

        X = df[self.feature_columns]
        y = df['Aggregate rating']

        # Drop any rows with NaN values that might have been created by the coerce
        X = X.dropna()
        y = y.loc[X.index]

        return X, y

    def train(self, data_path, n_estimators=100, max_depth=10, refit_encoders=True):
        """Trains the RandomForestRegressor model."""
        try:
            df = pd.read_csv(data_path)
            
            # For this dataset, 'Restaurant ID' is not a feature for prediction
            df.rename(columns={
                'Has Table booking': 'Has Table booking',
                'Has Online delivery': 'Has Online delivery',
                'Average Cost for two': 'Cost',
                'Restaurant Type': 'Rest type'
            }, inplace=True)
            
            # Handle string-based bool columns and convert to int
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
                n_jobs=-1  # Use all available cores for training
            )
            self.model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model trained with n_estimators={n_estimators}, max_depth={max_depth}")
            print(f"Mean Squared Error (MSE): {mse}")
            print(f"R-squared (R2): {r2}")

            # Save the trained model and encoders
            joblib.dump(self.model, 'models/rating_predictor.pkl')
            joblib.dump(self.encoders, 'models/rating_encoders.pkl')
            joblib.dump(self.feature_columns, 'models/rating_features.pkl')

            return {"mse": mse, "r2": r2}

        except Exception as e:
            print(f"An error occurred during training: {e}")
            return {"error": str(e)}

    def load_model(self):
        """Loads a pre-trained model and its encoders."""
        try:
            self.model = joblib.load('models/rating_predictor.pkl')
            self.encoders = joblib.load('models/rating_encoders.pkl')
            self.feature_columns = joblib.load('models/rating_features.pkl')
            print("Pre-trained model, encoders, and features loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}. Please ensure you have run the training script first.")
            self.model = None

    def _create_cache_key(self, data: dict) -> str:
        """Generates a consistent cache key from input data."""
        # Use a sorted tuple of (key, value) pairs to ensure consistency
        sorted_items = sorted(data.items())
        return json.dumps(sorted_items)

    def predict_rating(self, data: dict):
        """
        Predicts the rating for a given restaurant using caching.
        This function is designed to be called by FastAPI endpoints.
        """
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

        # Convert input data to a DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Preprocess the input data using the loaded encoders
        for col in self.encoders:
            if col in df.columns:
                # Handle unseen labels by mapping them to a default value (e.g., -1)
                le = self.encoders[col]
                # Filter out unseen labels
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                
                # Update the encoder's classes to include 'Unknown' if not present
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                
                # Transform using the updated encoder
                df[col] = le.transform(df[col])
            else:
                # If a feature is missing, fill with a default value (e.g., 0)
                df[col] = 0

        # Ensure feature order matches the trained model
        input_data = df[self.feature_columns].values
        
        # Make a prediction
        prediction = self.model.predict(input_data)[0]
        
        # Clamp the prediction to a reasonable range, e.g., 0-5
        prediction = max(0.0, min(5.0, float(prediction)))

        if self.redis_client:
            self.redis_client.setex(cache_key, 3600, json.dumps(prediction)) # Cache for 1 hour

        return prediction

    def objective(self, trial, data_path='data/Dataset.csv'):
        """Objective function for Optuna hyperparameter tuning."""
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        
        # Load and preprocess data (use refit_encoders=True for each trial)
        try:
            df = pd.read_csv(data_path)
            
            # Clean and prepare data
            df.rename(columns={
                'Has Table booking': 'Has Table booking',
                'Has Online delivery': 'Has Online delivery',
                'Average Cost for two': 'Cost',
                'Restaurant Type': 'Rest type'
            }, inplace=True)
            df['Has Table booking'] = df['Has Table booking'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
            df['Has Online delivery'] = df['Has Online delivery'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)

            X, y = self.preprocess_data(df, refit_encoders=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a model for this trial
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            model.fit(X_train, y_train)

            # Evaluate and return the score
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            return r2
            
        except FileNotFoundError:
            print(f"Data file not found at {data_path}")
            return -1.0 # Return a bad score to indicate an issue
        except Exception as e:
            print(f"An error occurred during tuning trial: {e}")
            return -1.0
            
    def tune_hyperparameters(self, n_trials=50, data_path='data/Dataset.csv'):
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
            
        # Retrain with best hyperparameters
        best_params = study.best_params
        print("Retraining model with best hyperparameters...")
        
        # Use the same encoders, don't refit them
        results = self.train(
            data_path=data_path,
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            refit_encoders=False
        )
        print("Model retrained and saved successfully!")
        return results


if __name__ == "__main__":
    predictor = RatingPredictor()
    
    # IMPORTANT: Check your actual filename - remove space if it exists
    # It should be either "Dataset.csv" or "Dataset .csv" (with space)
    DATA_PATH = "data/Dataset .csv"  
    
    # Check if file exists and suggest correction
    if not os.path.exists(DATA_PATH):
        print(f"File not found at: {DATA_PATH}")
        # Try alternative path without space
        alt_path = "data/Dataset.csv"
        if os.path.exists(alt_path):
            print(f"Found file at: {alt_path}")
            DATA_PATH = alt_path
        else:
            print("Please verify the correct file path and name.")
            exit(1)
    
    # --- Option 1: Train with default hyperparameters ---
    # results = predictor.train(DATA_PATH)
    # print("\nInitial Training Results:")
    # print(results)
    
    # --- Option 2: Tune hyperparameters and retrain ---
    results = predictor.tune_hyperparameters(n_trials=20, data_path=DATA_PATH)
    print("\nHyperparameter Tuning Results:")
    print(results)
    
    # --- Example prediction ---
    # Load the best model
    predictor.load_model()
    
    # Example data for prediction
    new_data = {
        'online_order': 1,
        'book_table': 1,
        'votes': 100,
        'location': 'Koramangala 5th Block',
        'rest_type': 'Casual Dining',
        'cuisines': 'North Indian, Chinese',
        'cost': 800
    }
    
    try:
        predicted_rating = predictor.predict_rating(new_data)
        print(f"\nPredicted rating for the example data: {predicted_rating:.2f}")
    except RuntimeError as e:
        print(e)