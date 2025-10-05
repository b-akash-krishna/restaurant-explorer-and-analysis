import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

class RestaurantRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.model_dir = os.path.join(BASE_DIR, 'models')
        self.data_path = os.path.join(BASE_DIR, 'data', 'Dataset.csv')

    def load_data(self):
        """Load and preprocess restaurant data"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['Cuisines'] = self.df['Cuisines'].fillna('Unknown')
            self.df['City'] = self.df['City'].fillna('Unknown')
            self.df['Price range'] = self.df['Price range'].fillna(0)
            self.df['Has Table booking'] = self.df['Has Table booking'].fillna('No')
            self.df['Has Online delivery'] = self.df['Has Online delivery'].fillna('No')
            
            self.df['combined_features'] = (
                self.df['Cuisines'].astype(str) + ' ' +
                self.df['City'].astype(str) + ' ' +
                'price_' + self.df['Price range'].astype(str) + ' ' +
                'table_' + self.df['Has Table booking'].astype(str).str.lower() + ' ' +
                'delivery_' + self.df['Has Online delivery'].astype(str).str.lower()
            )
    
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            self.df = None
            self.tfidf_matrix = None
            self.vectorizer = None

    def recommend(self, cuisine=None, city=None, price_range=None, table_booking=None, online_delivery=None, top_n=10):
        """Recommend restaurants based on user preferences"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return []

        filtered_df = self.df.copy()
        
        if city:
            filtered_df = filtered_df[filtered_df['City'].str.lower() == city.lower()]
        
        if cuisine:
            filtered_df = filtered_df[filtered_df['Cuisines'].str.contains(cuisine, case=False, na=False)]
        
        if price_range is not None:
            filtered_df = filtered_df[filtered_df['Price range'] == price_range]
        
        if table_booking is not None:
            booking_value = 'Yes' if table_booking else 'No'
            filtered_df = filtered_df[filtered_df['Has Table booking'].str.lower() == booking_value.lower()]
        
        if online_delivery is not None:
            delivery_value = 'Yes' if online_delivery else 'No'
            filtered_df = filtered_df[filtered_df['Has Online delivery'].str.lower() == delivery_value.lower()]
        
        if len(filtered_df) == 0:
            return []
        
        query_parts = []
        if cuisine:
            query_parts.append(cuisine)
        if city:
            query_parts.append(city)
        if price_range is not None:
            query_parts.append(f"price_{price_range}")
        if table_booking is not None:
            query_parts.append(f"table_{'yes' if table_booking else 'no'}")
        if online_delivery is not None:
            query_parts.append(f"delivery_{'yes' if online_delivery else 'no'}")
        
        query_string = ' '.join(query_parts)
        
        if query_string:
            query_vec = self.vectorizer.transform([query_string])
            filtered_indices = filtered_df.index.tolist()
            filtered_tfidf = self.tfidf_matrix[filtered_indices]
            cosine_sim = cosine_similarity(query_vec, filtered_tfidf).flatten()
            
            top_indices = cosine_sim.argsort()[-min(top_n, len(filtered_df)):][::-1]
            result_df = filtered_df.iloc[top_indices]
            similarities = cosine_sim[top_indices]
        else:
            result_df = filtered_df.head(top_n)
            similarities = np.ones(len(result_df))
        
        recommendations = []
        for idx, (_, row) in enumerate(result_df.iterrows()):
            recommendations.append({
                'name': row.get('Restaurant Name', 'Unknown'),
                'cuisine': row.get('Cuisines', 'Unknown'),
                'city': row.get('City', 'Unknown'),
                'rating': float(row.get('Aggregate rating', 0)),
                'price_range': int(row.get('Price range', 0)),
                'cost': int(row.get('Average Cost for two', 0)),
                'votes': int(row.get('Votes', 0)),
                'online_delivery': str(row.get('Has Online delivery', 'No')),
                'table_booking': str(row.get('Has Table booking', 'No')),
                'similarity_score': float(similarities[idx]) if idx < len(similarities) else 0.0
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
    
        return sorted(list(cuisines))

    def get_unique_cities(self):
        """Get list of unique cities"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return []
        return sorted(self.df['City'].dropna().unique().tolist())

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
            print(f"Error loading model files: {e}")
            self.vectorizer = None
            self.df = None
            self.tfidf_matrix = None