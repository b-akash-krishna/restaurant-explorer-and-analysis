import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib
import os

class CuisineClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_encoders = {}
        self.feature_columns = []

    def preprocess_data(self, df):
        """Preprocess data for cuisine classification"""
        df = df.copy()

        df['Cuisines'] = df['Cuisines'].fillna('Unknown')
        df['main_cuisine'] = df['Cuisines'].apply(lambda x: str(x).split(',')[0].strip())

        cuisine_counts = df['main_cuisine'].value_counts()
        top_cuisines = cuisine_counts.head(10).index.tolist()
        df = df[df['main_cuisine'].isin(top_cuisines)]

        categorical_cols = ['City', 'Has Table booking', 'Has Online delivery']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.feature_encoders:
                    self.feature_encoders[col] = LabelEncoder()
                    df[col] = df[col].fillna('Unknown')
                    self.feature_encoders[col].fit(df[col].astype(str))
                df[col] = self.feature_encoders[col].transform(df[col].astype(str))

        numeric_cols = ['Aggregate rating', 'Votes', 'Price range', 'Average Cost for two']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def train(self, data_path):
        """Train the cuisine classification model"""
        df = pd.read_csv(data_path)

        df = self.preprocess_data(df)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['main_cuisine'])

        self.feature_columns = ['Aggregate rating', 'Votes', 'Price range', 'Average Cost for two', 'Has Table booking', 'Has Online delivery']
        available_features = [col for col in self.feature_columns if col in df.columns]

        X = df[available_features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        cuisine_performance = {}
        for idx, cuisine in enumerate(self.label_encoder.classes_):
            mask = y_test == idx
            if mask.sum() > 0:
                cuisine_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                cuisine_performance[cuisine] = float(cuisine_accuracy)

        self.save_model()

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'cuisine_performance': cuisine_performance,
            'cuisines': self.label_encoder.classes_.tolist()
        }

    def predict(self, features):
        """Predict cuisine for given features"""
        if self.model is None:
            self.load_model()

        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]

        cuisine = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        return {
            'cuisine': cuisine,
            'confidence': confidence
        }

    def save_model(self):
        """Save trained model and encoders"""
        os.makedirs('backend/models', exist_ok=True)
        joblib.dump(self.model, 'backend/models/cuisine_classifier.pkl')
        joblib.dump(self.label_encoder, 'backend/models/cuisine_label_encoder.pkl')
        joblib.dump(self.feature_encoders, 'backend/models/cuisine_feature_encoders.pkl')
        joblib.dump(self.feature_columns, 'backend/models/cuisine_features.pkl')

    def load_model(self):
        """Load trained model and encoders"""
        try:
            self.model = joblib.load('backend/models/cuisine_classifier.pkl')
            self.label_encoder = joblib.load('backend/models/cuisine_label_encoder.pkl')
            self.feature_encoders = joblib.load('backend/models/cuisine_feature_encoders.pkl')
            self.feature_columns = joblib.load('backend/models/cuisine_features.pkl')
        except:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
