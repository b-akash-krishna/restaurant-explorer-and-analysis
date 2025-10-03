import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import optuna


class RatingPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = []

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
                    known_classes = set(self.encoders[col].classes_)
                    
                    # Map unseen labels to 'Unknown' if it exists in classes
                    if 'Unknown' in known_classes:
                        df[col] = df[col].apply(
                            lambda x: x if x in known_classes else 'Unknown'
                        )
                    
                    df[col] = self.encoders[col].transform(df[col])

        numeric_cols = ['Votes', 'Average Cost for two']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def train(self, data_path, n_estimators=100, max_depth=None, refit_encoders=True):
        """Train the model with a given set of hyperparameters and evaluate."""
        try:
            df = pd.read_csv(data_path)
            print(f"Dataset loaded successfully: {len(df)} rows")
        except FileNotFoundError:
            print(f"Error: Dataset not found at {data_path}")
            return None

        processed_df = self.preprocess_data(df, fit_encoders=refit_encoders)
        print(f"After preprocessing: {len(processed_df)} rows")

        available_features = [
            'Votes', 'Average Cost for two', 'Price range', 'Has Table booking', 'Has Online delivery'
        ]

        # Safety check: ensure columns exist before slicing
        missing_cols = [col for col in available_features if col not in processed_df.columns]
        if missing_cols:
            print(f"Error: Missing required feature columns: {missing_cols}")
            return None

        X = processed_df[available_features]
        y = processed_df['Aggregate rating']

        # Check if we have valid data
        if len(X) == 0 or X.shape[1] == 0:
            print("Error: No valid features after preprocessing")
            return None

        self.feature_columns = available_features

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")

        self.save_model(n_estimators, max_depth)

        return {
            'mse': float(mse),
            'r2': float(r2),
            'feature_importance': dict(zip(available_features, self.model.feature_importances_.tolist()))
        }

    def predict(self, features):
        """Predict rating for given features"""
        if self.model is None:
            self.load_model()

        prediction = self.model.predict([features])[0]
        return float(max(0, min(5, prediction)))

    def save_model(self, n_estimators, max_depth):
        """Save trained model and encoders"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/rating_predictor.pkl')
        joblib.dump(self.encoders, 'models/rating_encoders.pkl')
        joblib.dump(self.feature_columns, 'models/rating_features.pkl')
        
        # Save tuning results
        tuning_results = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        joblib.dump(tuning_results, 'models/rating_tuning_results.pkl')
        print(f"Model saved successfully")

    def load_model(self):
        """Load trained model and encoders"""
        try:
            self.model = joblib.load('backend/models/rating_predictor.pkl')
            self.encoders = joblib.load('backend/models/rating_encoders.pkl')
            self.feature_columns = joblib.load('backend/models/rating_features.pkl')
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Model not found. Please train the model first.")
            self.model = None

    def tune_hyperparameters(self, data_path, n_trials=50):
        """Perform hyperparameter tuning for the Random Forest model."""
        
        # Load and preprocess data once
        try:
            df = pd.read_csv(data_path)
            print(f"Dataset loaded for tuning: {len(df)} rows")
        except FileNotFoundError:
            print(f"Error: Dataset not found at {data_path}")
            return None

        # Fit encoders on the full dataset
        processed_df = self.preprocess_data(df, fit_encoders=True)
        print(f"After preprocessing for tuning: {len(processed_df)} rows")

        # Define feature columns
        available_features = [
            'Votes', 'Average Cost for two', 'Price range', 'Has Table booking', 'Has Online delivery'
        ]

        # Verify all features exist
        missing_cols = [col for col in available_features if col not in processed_df.columns]
        if missing_cols:
            print(f"Error: Missing required feature columns: {missing_cols}")
            return None

        self.feature_columns = available_features
        X = processed_df[self.feature_columns]
        y = processed_df['Aggregate rating']

        # Check if we have valid data
        if len(X) == 0 or X.shape[1] == 0:
            print("Error: No valid features after preprocessing")
            return None

        # Split data once for all trials
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 20)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Minimize Mean Squared Error
            mse = mean_squared_error(y_test, y_pred)
            return mse

        # Create a study object and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print("\n" + "="*50)
        print("Optimization finished.")
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best MSE: {study.best_value:.4f}")
        print("="*50 + "\n")

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
    # print("\nTraining Results:")
    # print(results)

    # --- Option 2: Tune hyperparameters (Recommended) ---
    predictor.tune_hyperparameters(DATA_PATH, n_trials=50)