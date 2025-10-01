import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class RatingPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = []

    def preprocess_data(self, df):
        """Preprocess restaurant data for rating prediction"""
        df = df.copy()

        df = df.dropna(subset=['Aggregate rating'])

        categorical_cols = ['City', 'Cuisines', 'Has Table booking', 'Has Online delivery', 'Price range']

        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = df[col].fillna('Unknown')
                    self.encoders[col].fit(df[col].astype(str))
                df[col] = self.encoders[col].transform(df[col].astype(str))

        numeric_cols = ['Votes', 'Average Cost for two']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def train(self, data_path):
        """Train the rating prediction model"""
        df = pd.read_csv(data_path)

        df = self.preprocess_data(df)

        self.feature_columns = ['Votes', 'Average Cost for two', 'Price range', 'Has Table booking', 'Has Online delivery']
        available_features = [col for col in self.feature_columns if col in df.columns]

        X = df[available_features]
        y = df['Aggregate rating']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.save_model()

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

    def save_model(self):
        """Save trained model and encoders"""
        os.makedirs('backend/models', exist_ok=True)
        joblib.dump(self.model, 'backend/models/rating_predictor.pkl')
        joblib.dump(self.encoders, 'backend/models/rating_encoders.pkl')
        joblib.dump(self.feature_columns, 'backend/models/rating_features.pkl')

    def load_model(self):
        """Load trained model and encoders"""
        try:
            self.model = joblib.load('backend/models/rating_predictor.pkl')
            self.encoders = joblib.load('backend/models/rating_encoders.pkl')
            self.feature_columns = joblib.load('backend/models/rating_features.pkl')
        except:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
