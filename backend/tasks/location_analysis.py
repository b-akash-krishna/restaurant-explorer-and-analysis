import pandas as pd
import numpy as np
import joblib
import os

class LocationAnalyzer:
    def __init__(self):
        self.df = None

    def load_data(self, data_path):
        """Load and preprocess location data"""
        self.df = pd.read_csv(data_path)

        self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
        self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
        self.df['City'] = self.df['City'].fillna('Unknown')
        self.df['Locality'] = self.df['Locality'].fillna('Unknown')

    def get_location_distribution(self):
        """Get distribution of restaurants by location"""
        if self.df is None:
            return []

        locations = []
        for idx, row in self.df.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                locations.append({
                    'lat': float(row['Latitude']),
                    'lng': float(row['Longitude']),
                    'name': row.get('Restaurant Name', 'Unknown'),
                    'city': row.get('City', 'Unknown'),
                    'rating': float(row.get('Aggregate rating', 0))
                })

        return locations[:500]

    def analyze_by_city(self):
        """Analyze restaurant statistics by city"""
        if self.df is None:
            return []

        city_stats = []
        for city in self.df['City'].value_counts().head(20).index:
            city_df = self.df[self.df['City'] == city]

            stats = {
                'city': city,
                'count': int(len(city_df)),
                'avg_rating': float(city_df['Aggregate rating'].mean()),
                'avg_cost': float(city_df['Average Cost for two'].mean()) if 'Average Cost for two' in city_df.columns else 0,
                'top_cuisine': city_df['Cuisines'].mode()[0] if len(city_df['Cuisines'].mode()) > 0 else 'Unknown'
            }
            city_stats.append(stats)

        return city_stats

    def analyze_by_locality(self, city=None):
        """Analyze restaurant statistics by locality"""
        if self.df is None:
            return []

        df_filtered = self.df if city is None else self.df[self.df['City'] == city]

        locality_stats = []
        for locality in df_filtered['Locality'].value_counts().head(15).index:
            locality_df = df_filtered[df_filtered['Locality'] == locality]

            stats = {
                'locality': locality,
                'city': locality_df['City'].mode()[0] if len(locality_df['City'].mode()) > 0 else 'Unknown',
                'count': int(len(locality_df)),
                'avg_rating': float(locality_df['Aggregate rating'].mean()),
                'price_range_mode': int(locality_df['Price range'].mode()[0]) if len(locality_df['Price range'].mode()) > 0 else 0
            }
            locality_stats.append(stats)

        return locality_stats

    def get_insights(self):
        """Get interesting insights from location data"""
        if self.df is None:
            return {}

        insights = {
            'total_restaurants': int(len(self.df)),
            'total_cities': int(self.df['City'].nunique()),
            'total_localities': int(self.df['Locality'].nunique()),
            'highest_rated_city': self.df.groupby('City')['Aggregate rating'].mean().idxmax(),
            'highest_restaurant_density_city': self.df['City'].value_counts().index[0],
            'avg_rating_overall': float(self.df['Aggregate rating'].mean()),
            'most_expensive_city': self.df.groupby('City')['Average Cost for two'].mean().idxmax() if 'Average Cost for two' in self.df.columns else 'N/A'
        }

        return insights
