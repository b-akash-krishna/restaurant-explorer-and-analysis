import pandas as pd
import numpy as np
import joblib
import os

# Corrected path to point to the backend directory
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Dataset.csv')

class LocationAnalyzer:
    def __init__(self):
        self.df = None

    def load_data(self, data_path=DATA_PATH):
        """Load and preprocess location data"""
        try:
            self.df = pd.read_csv(data_path)

            self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
            self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
            self.df['City'] = self.df['City'].fillna('Unknown')
            self.df['Locality'] = self.df['Locality'].fillna('Unknown')
        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Please ensure the file is in the correct location.")
            self.df = None

    def get_location_distribution(self):
        """Get distribution of restaurants by location"""
        if self.df is None:
            # Attempt to load data if it's not already loaded
            self.load_data()
            if self.df is None:
                return []

        locations = []
        # Limiting to 500 for demonstration to avoid large payloads
        for idx, row in self.df.head(500).iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                locations.append({
                    'lat': float(row['Latitude']),
                    'lng': float(row['Longitude']),
                    'name': row.get('Restaurant Name', 'Unknown'),
                    'city': row.get('City', 'Unknown'),
                    'rating': float(row.get('Aggregate rating', 0))
                })

        return locations

    def analyze_by_city(self):
        """Analyze restaurant distribution by city"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return {}

        city_stats = self.df.groupby('City').agg(
            restaurant_count=('Restaurant Name', 'size'),
            average_rating=('Aggregate rating', 'mean'),
            online_order_percentage=('Has Online delivery', lambda x: (x == 'Yes').mean() * 100),
            table_booking_percentage=('Has Table booking', lambda x: (x == 'Yes').mean() * 100)
        ).to_dict('index')

        return city_stats

    def analyze_by_locality(self, city):
        """Analyze localities within a specific city"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return []

        df_filtered = self.df[self.df['City'].str.lower() == city.lower()]
        if df_filtered.empty:
            return []

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
            self.load_data()
            if self.df is None:
                return {}

        insights = {
            'total_restaurants': int(len(self.df)),
            'total_cities': int(self.df['City'].nunique()),
            'total_localities': int(self.df['Locality'].nunique()),
            'highest_rated_city': self.df.groupby('City')['Aggregate rating'].mean().idxmax(),
            'highest_restaurant_density_city': self.df['City'].value_counts().index[0],
            'avg_rating_overall': float(self.df['Aggregate rating'].mean()),
            'most_expensive_city': self.df.groupby('City')['Average Cost for two'].mean().idxmax()
        }
        return insights


# Example usage (for testing)
if __name__ == "__main__":
    analyzer = LocationAnalyzer()
    analyzer.load_data()

    if analyzer.df is not None:
        print("Locations distribution (first 5):", analyzer.get_location_distribution()[:5])
        print("\nCity analysis:", analyzer.analyze_by_city())
        print("\nLocality analysis for New Delhi:", analyzer.analyze_by_locality('New Delhi'))
        print("\nGeneral insights:", analyzer.get_insights())