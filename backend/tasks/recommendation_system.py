import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class RestaurantRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = None

    def load_data(self, data_path):
        """Load and preprocess restaurant data"""
        self.df = pd.read_csv(data_path)

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

    def recommend(self, cuisine=None, city=None, price_range=None, top_n=10):
        """Recommend restaurants based on user preferences"""
        if self.df is None:
            return []

        filtered_df = self.df.copy()

        if cuisine and cuisine.lower() != 'any':
            filtered_df = filtered_df[filtered_df['Cuisines'].str.contains(cuisine, case=False, na=False)]

        if city and city.lower() != 'any':
            filtered_df = filtered_df[filtered_df['City'].str.contains(city, case=False, na=False)]

        if price_range is not None and price_range > 0:
            filtered_df = filtered_df[filtered_df['Price range'] == price_range]

        if len(filtered_df) == 0:
            filtered_df = self.df.copy()

        filtered_df = filtered_df.sort_values('Aggregate rating', ascending=False)

        recommendations = []
        for idx, row in filtered_df.head(top_n).iterrows():
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
            return []

        cuisines = set()
        for cuisine_str in self.df['Cuisines'].dropna():
            cuisines.update([c.strip() for c in str(cuisine_str).split(',')])

        return sorted(list(cuisines))[:50]

    def get_unique_cities(self):
        """Get list of unique cities"""
        if self.df is None:
            return []

        return sorted(self.df['City'].dropna().unique().tolist())[:50]

    def save_model(self):
        """Save recommendation data"""
        os.makedirs('backend/models', exist_ok=True)
        joblib.dump(self.vectorizer, 'backend/models/recommender_vectorizer.pkl')

    def load_model(self):
        """Load recommendation data"""
        try:
            self.vectorizer = joblib.load('backend/models/recommender_vectorizer.pkl')
        except:
            pass
