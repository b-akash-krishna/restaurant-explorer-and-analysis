import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib
import os
import requests
from bs4 import BeautifulSoup

# Corrected path to point to the backend directory
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Dataset.csv')

class CuisineClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_encoders = {}
        self.feature_columns = []
        # Define the base directory for model files
        self.model_dir = os.path.join(BASE_DIR, 'models')

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

        # Ensure all columns are numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        X = df[['City', 'Has Table booking', 'Has Online delivery', 'Price range', 'Votes']]
        y = df['main_cuisine']

        self.feature_columns = X.columns.tolist()

        # Fit the label encoder for the target variable
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        return X, y

    def train(self, data_path=DATA_PATH):
        """Trains the RandomForestClassifier model."""
        try:
            df = pd.read_csv(data_path)

            df.rename(columns={
                'Has Table booking': 'Has Table booking',
                'Has Online delivery': 'Has Online delivery',
                'Restaurant Type': 'Rest type',
                'Cuisines': 'Cuisines'
            }, inplace=True)
            
            # Handle string-based bool columns and convert to int
            if 'Has Table booking' in df.columns:
                df['Has Table booking'] = df['Has Table booking'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
            if 'Has Online delivery' in df.columns:
                df['Has Online delivery'] = df['Has Online delivery'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)

            X, y = self.preprocess_data(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
            
            print(f"Accuracy: {accuracy}")
            print("Classification Report:\n", report)

            self.save_model()
            return {"accuracy": accuracy, "report": report}
        
        except FileNotFoundError as e:
            print(f"Error: Data file not found at {data_path}. {e}")
            return {"error": str(e)}

    def predict(self, features):
        """Predict cuisine for given features"""
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise RuntimeError("Model is not loaded. Please train or load a model first.")

        # Ensure features are in the correct order for the model
        features_df = pd.DataFrame([features])
        for col, encoder in self.feature_encoders.items():
            if col in features_df.columns:
                features_df[col] = features_df[col].apply(
                    lambda x: x if x in encoder.classes_ else 'Unknown'
                )
                if 'Unknown' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, 'Unknown')
                features_df[col] = encoder.transform(features_df[col].astype(str))
        
        features_array = features_df[self.feature_columns].values
        
        prediction = self.model.predict(features_array)[0]
        cuisine = self.label_encoder.inverse_transform([prediction])[0]
        
        return cuisine

    def classify_from_url(self, url: str):
        """
        Fetches text from a given URL and classifies the cuisine.
        This is a placeholder for a more complex web scraping and NLP task.
        """
        try:
            # Step 1: Fetch the webpage
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raises an HTTPError for bad responses

            # Step 2: Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Step 3: Extract text (this is a simplified approach)
            text_content = soup.get_text(separator=' ', strip=True)

            # Step 4: A simple keyword-based classification
            # This is a placeholder for a more sophisticated ML model
            cuisine_keywords = {
                'Italian': ['pasta', 'pizza', 'risotto', 'lasagna'],
                'Mexican': ['taco', 'burrito', 'quesadilla', 'nachos'],
                'Chinese': ['kung pao', 'chow mein', 'dumplings', 'fried rice'],
                'Indian': ['curry', 'naan', 'tandoori', 'biryani'],
                'Japanese': ['sushi', 'ramen', 'teriyaki', 'sashimi']
            }
            
            text_lower = text_content.lower()
            
            for cuisine, keywords in cuisine_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    return cuisine

            return "Unknown"

        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            raise RuntimeError(f"Could not fetch data from URL: {url}")
        except Exception as e:
            print(f"Error during classification: {e}")
            raise RuntimeError("An error occurred during cuisine classification.")


    def save_model(self):
        """Save trained model and encoders"""
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(self.model_dir, 'cuisine_classifier.pkl'))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, 'cuisine_label_encoder.pkl'))
        joblib.dump(self.feature_encoders, os.path.join(self.model_dir, 'cuisine_feature_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(self.model_dir, 'cuisine_features.pkl'))

    def load_model(self):
        """Load trained model and encoders"""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'cuisine_classifier.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_dir, 'cuisine_label_encoder.pkl'))
            self.feature_encoders = joblib.load(os.path.join(self.model_dir, 'cuisine_feature_encoders.pkl'))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, 'cuisine_features.pkl'))
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}. Please ensure you have run the training script first.")
            self.model = None

# Example usage (for testing)
if __name__ == '__main__':
    classifier = CuisineClassifier()
    # First, train the model if it doesn't exist
    if not os.path.exists(os.path.join(classifier.model_dir, 'cuisine_classifier.pkl')):
        print("Model not found, training new model...")
        classifier.train()
    else:
        classifier.load_model()

    # Example prediction from URL
    try:
        url_to_classify = 'https://www.allrecipes.com/recipes/723/world-cuisine/asian/chinese/main-dishes/'
        print(f"Classifying cuisine from URL: {url_to_classify}")
        cuisine = classifier.classify_from_url(url_to_classify)
        print(f"The cuisine for the URL is: {cuisine}")
    except RuntimeError as e:
        print(e)