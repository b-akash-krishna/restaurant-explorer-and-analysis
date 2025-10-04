import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import os
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Dataset.csv')

class CuisineClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_encoders = {}
        self.feature_columns = []
        self.model_dir = os.path.join(BASE_DIR, 'models')
        self.evaluation_metrics = {}
        self.per_cuisine_metrics = {}
        self.class_distribution = {}

    def preprocess_data(self, df):
        """Preprocess data for cuisine classification"""
        df = df.copy()

        # Handle cuisines
        df['Cuisines'] = df['Cuisines'].fillna('Unknown')
        df['main_cuisine'] = df['Cuisines'].apply(lambda x: str(x).split(',')[0].strip())

        # Analyze class distribution
        cuisine_counts = df['main_cuisine'].value_counts()
        self.class_distribution = cuisine_counts.to_dict()
        
        print(f"\nClass Distribution Analysis:")
        print(f"Total unique cuisines: {len(cuisine_counts)}")
        print(f"Top 10 cuisines by frequency:")
        for cuisine, count in cuisine_counts.head(10).items():
            print(f"  {cuisine}: {count} ({count/len(df)*100:.2f}%)")
        
        # Filter to top cuisines
        top_cuisines = cuisine_counts.head(10).index.tolist()
        df = df[df['main_cuisine'].isin(top_cuisines)]
        
        print(f"\nUsing top 10 cuisines for classification")
        print(f"Total samples after filtering: {len(df)}")

        # Select and prepare features - DON'T convert everything to numeric yet!
        feature_cols = ['City', 'Has Table booking', 'Has Online delivery', 'Price range', 'Votes']
        
        # Ensure all feature columns exist and handle missing values
        for col in feature_cols:
            if col not in df.columns:
                print(f"Warning: Column {col} not found in dataset")
                df[col] = 'Unknown' if col in ['City'] else 0
            else:
                if col == 'City':
                    df[col] = df[col].fillna('Unknown')
                elif col in ['Has Table booking', 'Has Online delivery']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(0)
        
        # Encode categorical features BEFORE converting to numeric
        categorical_cols = ['City', 'Has Table booking', 'Has Online delivery']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
                if col not in self.feature_encoders:
                    self.feature_encoders[col] = LabelEncoder()
                    self.feature_encoders[col].fit(df[col])
                
                df[col] = self.feature_encoders[col].transform(df[col])
        
        # Now ensure numeric types for numeric columns
        for col in ['Price range', 'Votes']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Select features and target
        X = df[feature_cols]
        y = df['main_cuisine']
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset size after preprocessing: {len(X)}")
        
        if len(X) == 0:
            raise ValueError("No samples remaining after preprocessing! Check your data.")
        
        self.feature_columns = feature_cols
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        return X, y

    def train(self, data_path=DATA_PATH):
        """Trains the RandomForestClassifier model with comprehensive evaluation."""
        try:
            df = pd.read_csv(data_path)

            # Rename columns to match expected names
            df.rename(columns={
                'Has Table booking': 'Has Table booking',
                'Has Online delivery': 'Has Online delivery',
                'Restaurant Type': 'Rest type',
                'Cuisines': 'Cuisines'
            }, inplace=True)
            
            # Convert Yes/No to 1/0 for boolean columns
            if 'Has Table booking' in df.columns:
                df['Has Table booking'] = df['Has Table booking'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
            if 'Has Online delivery' in df.columns:
                df['Has Online delivery'] = df['Has Online delivery'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

            X, y = self.preprocess_data(df)
            
            if len(X) == 0:
                raise ValueError("No data available for training after preprocessing")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            print(f"\nTraining set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")

            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X_train, y_train)

            # Predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate comprehensive metrics
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'precision_macro': precision_score(y_train, y_pred_train, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'recall_macro': recall_score(y_train, y_pred_train, average='macro', zero_division=0),
                'recall_weighted': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_train, y_pred_train, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            }
            
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'precision_macro': precision_score(y_test, y_pred_test, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall_macro': recall_score(y_test, y_pred_test, average='macro', zero_division=0),
                'recall_weighted': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_test, y_pred_test, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            }
            
            self.evaluation_metrics = {
                'train': train_metrics,
                'test': test_metrics
            }

            # Per-cuisine performance analysis
            report = classification_report(
                y_test, y_pred_test,
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
            
            self.per_cuisine_metrics = {}
            for cuisine in self.label_encoder.classes_:
                if cuisine in report:
                    self.per_cuisine_metrics[cuisine] = {
                        'precision': report[cuisine]['precision'],
                        'recall': report[cuisine]['recall'],
                        'f1-score': report[cuisine]['f1-score'],
                        'support': report[cuisine]['support']
                    }

            # Confusion matrix for bias analysis
            cm = confusion_matrix(y_test, y_pred_test)
            
            print(f"\n{'='*60}")
            print(f"Cuisine Classification Training Complete")
            print(f"{'='*60}")
            
            print(f"\nTraining Set Metrics:")
            for metric, value in train_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nTest Set Metrics:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nPer-Cuisine Performance (Test Set):")
            print(f"{'Cuisine':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
            print(f"{'-'*70}")
            for cuisine, metrics in self.per_cuisine_metrics.items():
                print(f"{cuisine:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                      f"{metrics['f1-score']:<12.4f} {int(metrics['support'])}")
            
            # Identify challenges and biases
            print(f"\n{'='*60}")
            print(f"Bias and Challenge Analysis")
            print(f"{'='*60}")
            
            # Find poorly performing cuisines
            poor_performers = [(c, m['f1-score']) 
                             for c, m in self.per_cuisine_metrics.items() 
                             if m['f1-score'] < 0.7]
            poor_performers.sort(key=lambda x: x[1])
            
            if poor_performers:
                print(f"\nCuisines with F1-score < 0.7 (potential challenges):")
                for cuisine, score in poor_performers:
                    support = self.per_cuisine_metrics[cuisine]['support']
                    print(f"  {cuisine}: F1={score:.4f}, Support={int(support)}")
                    if support < 50:
                        print(f"    → Issue: Low sample size (class imbalance)")
                    if self.per_cuisine_metrics[cuisine]['precision'] < 0.6:
                        print(f"    → Issue: High false positive rate")
                    if self.per_cuisine_metrics[cuisine]['recall'] < 0.6:
                        print(f"    → Issue: Many samples misclassified as other cuisines")
            else:
                print(f"\nNo significant performance issues detected (all F1-scores >= 0.7)")
            
            # Check for class imbalance bias
            supports = [m['support'] for m in self.per_cuisine_metrics.values()]
            max_support = max(supports)
            min_support = min(supports)
            imbalance_ratio = max_support / min_support if min_support > 0 else float('inf')
            
            print(f"\nClass Imbalance Analysis:")
            print(f"  Largest class size: {int(max_support)}")
            print(f"  Smallest class size: {int(min_support)}")
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
            if imbalance_ratio > 5:
                print(f"  ⚠️  Warning: Significant class imbalance detected!")
                print(f"     Model may be biased toward majority classes.")
            
            print(f"\n{'='*60}\n")

            # Save everything
            self.save_model()
            
            return {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'per_cuisine_metrics': self.per_cuisine_metrics,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'biases_identified': len(poor_performers) > 0 or imbalance_ratio > 5
            }
        
        except FileNotFoundError as e:
            print(f"Error: Data file not found at {data_path}. {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def predict(self, features):
        """Predict cuisine for given features"""
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise RuntimeError("Model is not loaded. Please train or load a model first.")

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
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]
        
        cuisine = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities)) * 100
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'cuisine': self.label_encoder.inverse_transform([idx])[0],
                'confidence': float(probabilities[idx]) * 100
            }
            for idx in top_indices
        ]
        
        return {
            'predicted_cuisine': cuisine,
            'confidence': confidence,
            'top_predictions': top_predictions
        }

    def classify_from_url(self, url: str):
        """Fetches text from a given URL and classifies the cuisine."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)

            cuisine_keywords = {
                'Italian': ['pasta', 'pizza', 'risotto', 'lasagna', 'spaghetti', 'parmesan'],
                'Mexican': ['taco', 'burrito', 'quesadilla', 'nachos', 'salsa', 'guacamole'],
                'Chinese': ['kung pao', 'chow mein', 'dumplings', 'fried rice', 'wonton', 'dim sum'],
                'Indian': ['curry', 'naan', 'tandoori', 'biryani', 'masala', 'tikka'],
                'Japanese': ['sushi', 'ramen', 'teriyaki', 'sashimi', 'tempura', 'udon'],
                'North Indian': ['butter chicken', 'dal', 'paneer', 'roti', 'kulcha'],
                'South Indian': ['dosa', 'idli', 'sambar', 'vada', 'upma'],
                'Thai': ['pad thai', 'curry', 'tom yum', 'satay', 'coconut'],
                'Continental': ['steak', 'salad', 'soup', 'grilled', 'roasted']
            }
            
            text_lower = text_content.lower()
            
            # Count keyword matches
            matches = {}
            for cuisine, keywords in cuisine_keywords.items():
                matches[cuisine] = sum(1 for keyword in keywords if keyword in text_lower)
            
            # Return cuisine with most matches
            if max(matches.values()) > 0:
                best_cuisine = max(matches, key=matches.get)
                confidence = (matches[best_cuisine] / len(cuisine_keywords[best_cuisine])) * 100
                return {
                    'cuisine': best_cuisine,
                    'confidence': min(confidence, 95.0),
                    'method': 'keyword_matching'
                }

            return {
                'cuisine': 'Unknown',
                'confidence': 0.0,
                'method': 'keyword_matching'
            }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            raise RuntimeError(f"Could not fetch data from URL: {url}")
        except Exception as e:
            print(f"Error during classification: {e}")
            raise RuntimeError("An error occurred during cuisine classification.")

    def get_model_performance_summary(self):
        """Get a summary of model performance for API endpoints"""
        if not self.model:
            self.load_model()
        
        return {
            'test_metrics': self.evaluation_metrics.get('test', {}),
            'per_cuisine_metrics': self.per_cuisine_metrics,
            'total_cuisines': len(self.label_encoder.classes_) if self.label_encoder else 0,
            'cuisines': list(self.label_encoder.classes_) if self.label_encoder else []
        }

    def save_model(self):
        """Save trained model and encoders"""
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(self.model_dir, 'cuisine_classifier.pkl'))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, 'cuisine_label_encoder.pkl'))
        joblib.dump(self.feature_encoders, os.path.join(self.model_dir, 'cuisine_feature_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(self.model_dir, 'cuisine_features.pkl'))
        joblib.dump(self.evaluation_metrics, os.path.join(self.model_dir, 'cuisine_metrics.pkl'))
        joblib.dump(self.per_cuisine_metrics, os.path.join(self.model_dir, 'per_cuisine_metrics.pkl'))

    def load_model(self):
        """Load trained model and encoders"""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'cuisine_classifier.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_dir, 'cuisine_label_encoder.pkl'))
            self.feature_encoders = joblib.load(os.path.join(self.model_dir, 'cuisine_feature_encoders.pkl'))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, 'cuisine_features.pkl'))
            
            # Load metrics if available
            try:
                self.evaluation_metrics = joblib.load(os.path.join(self.model_dir, 'cuisine_metrics.pkl'))
                self.per_cuisine_metrics = joblib.load(os.path.join(self.model_dir, 'per_cuisine_metrics.pkl'))
            except FileNotFoundError:
                self.evaluation_metrics = {}
                self.per_cuisine_metrics = {}
                
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}. Please ensure you have run the training script first.")
            self.model = None


if __name__ == '__main__':
    classifier = CuisineClassifier()
    
    # Train the model with comprehensive evaluation
    if not os.path.exists(os.path.join(classifier.model_dir, 'cuisine_classifier.pkl')):
        print("Model not found, training new model...")
        results = classifier.train()
        if 'error' not in results:
            print("\n=== Training Summary ===")
            print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
            print(f"Biases Identified: {results['biases_identified']}")
    else:
        classifier.load_model()
        print("Model loaded successfully")

    # Example prediction
    try:
        example_features = {
            'City': 'Bangalore',
            'Has Table booking': 1,
            'Has Online delivery': 1,
            'Price range': 2,
            'Votes': 150
        }
        result = classifier.predict(example_features)
        print(f"\nExample Prediction:")
        print(f"Predicted Cuisine: {result['predicted_cuisine']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Top 3 Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['cuisine']}: {pred['confidence']:.2f}%")
    except Exception as e:
        print(f"Error in prediction: {e}")