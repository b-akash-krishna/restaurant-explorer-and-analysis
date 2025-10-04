import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Corrected path to point to the backend directory
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

class RestaurantRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        # Define the base directory for model and data files
        self.model_dir = os.path.join(BASE_DIR, 'models')
        self.data_path = os.path.join(BASE_DIR, 'data', 'Dataset.csv')

    def load_data(self):
        """Load and preprocess restaurant data"""
        try:
            self.df = pd.read_csv(self.data_path)

            self.df['Cuisines'] = self.df['Cuisines'].fillna('Unknown')
            self.df['City'] = self.df['City'].fillna('Unknown')
            self.df['Price range'] = self.df['Price range'].fillna(0)
    
            self.df['combined_features'] = (
                self.df['Cuisines'].astype(str) + ' ' +
                self.df['City'].astype(str) + ' ' +
                'price_' + self.df['Price range'].astype(str)
            )
    
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])
        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Please ensure the file is in the correct location.")
            self.df = None
            self.tfidf_matrix = None
            self.vectorizer = None

    def recommend(self, cuisine=None, city=None, price_range=None, top_n=10):
        """Recommend restaurants based on user preferences"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return []

        # Create a query string from user preferences
        query_parts = []
        if cuisine:
            query_parts.append(cuisine)
        if city:
            query_parts.append(city)
        if price_range:
            query_parts.append(f"price_{price_range}")
        
        query_string = ' '.join(query_parts)
        
        if not query_string:
            # If no preferences, return a default list or an empty list
            return []

        # Transform the query string to a TF-IDF vector
        query_vec = self.vectorizer.transform([query_string])
        
        # Compute cosine similarity between the query vector and all restaurant vectors
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get the indices of the top-N most similar restaurants
        similar_indices = cosine_sim.argsort()[-top_n:][::-1]
        
        # Get the recommended restaurants from the DataFrame
        recommendations = []
        for idx in similar_indices:
            row = self.df.iloc[idx]
            recommendations.append({
                'name': row.get('Restaurant Name', 'Unknown'),
                'cuisine': row.get('Cuisines', 'Unknown'),
                'city': row.get('City', 'Unknown'),
                'rating': float(row.get('Aggregate rating', 0)),
                'price_range': int(row.get('Price range', 0)),
                'votes': int(row.get('Votes', 0))
            })
    
        return recommendations

    def get_unique_cuisines(self):
        """Get list of unique cuisines"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return []
        
        cuisines = set()
        for cuisine_str in self.df['Cuisines'].dropna():
            cuisines.update([c.strip() for c in str(cuisine_str).split(',')])
    
        return sorted(list(cuisines))[:50]

    def get_unique_cities(self):
        """Get list of unique cities"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return []

        return sorted(self.df['City'].dropna().unique().tolist())[:50]

    def save_model(self):
        """Save recommendation data"""
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(self.model_dir, 'recommender_vectorizer.pkl'))
        joblib.dump(self.df, os.path.join(self.model_dir, 'recommender_df.pkl'))
        joblib.dump(self.tfidf_matrix, os.path.join(self.model_dir, 'recommender_tfidf_matrix.pkl'))

    def load_model(self):
        """Load trained model and encoders"""
        try:
            self.vectorizer = joblib.load(os.path.join(self.model_dir, 'recommender_vectorizer.pkl'))
            self.df = joblib.load(os.path.join(self.model_dir, 'recommender_df.pkl'))
            self.tfidf_matrix = joblib.load(os.path.join(self.model_dir, 'recommender_tfidf_matrix.pkl'))
            print("Recommender model, data, and vectorizer loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}. Please ensure you have run the training script first.")
            self.vectorizer = None
            self.df = None
            self.tfidf_matrix = None

# Example usage (for testing)
if __name__ == '__main__':
    recommender = RestaurantRecommender()
    recommender.load_data()

    if recommender.df is not None:
        print("Unique Cuisines:", recommender.get_unique_cuisines())
        print("Unique Cities:", recommender.get_unique_cities())
        
        # Example recommendation
        recommendations = recommender.recommend(cuisine='North Indian', city='Bangalore')
        print("Recommendations for North Indian restaurants in Bangalore:")
        for rec in recommendations:
            print(f"- {rec['name']} ({rec['cuisine']}, {rec['city']}) - Rating: {rec['rating']}")