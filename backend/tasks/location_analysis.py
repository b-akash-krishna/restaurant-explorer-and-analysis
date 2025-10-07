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
            
            # Ensure numeric columns
            self.df['Aggregate rating'] = pd.to_numeric(self.df['Aggregate rating'], errors='coerce').fillna(0)
            self.df['Average Cost for two'] = pd.to_numeric(self.df['Average Cost for two'], errors='coerce').fillna(0)
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Please ensure the file is in the correct location.")
            self.df = None

    def get_map_data(self):
        """Get location data for map visualization"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return []

        locations = []
        # Filter valid coordinates and limit to 500 for performance
        valid_df = self.df.dropna(subset=['Latitude', 'Longitude'])
        
        for idx, row in valid_df.head(500).iterrows():
            try:
                locations.append({
                    'lat': float(row['Latitude']),
                    'lng': float(row['Longitude']),
                    'name': str(row.get('Restaurant Name', 'Unknown')),
                    'city': str(row.get('City', 'Unknown')),
                    'rating': float(row.get('Aggregate rating', 0))
                })
            except (ValueError, TypeError):
                continue

        return locations

    def get_location_distribution(self):
        """Get distribution of restaurants by location"""
        if self.df is None:
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

        # Handle Yes/No values
        def convert_yes_no(x):
            if pd.isna(x):
                return 0
            return 1 if str(x).lower() == 'yes' else 0

        city_stats = {}
        for city in self.df['City'].unique():
            if pd.isna(city) or city == 'Unknown':
                continue
                
            city_df = self.df[self.df['City'] == city]
            
            city_stats[city] = {
                'restaurant_count': len(city_df),
                'average_rating': float(city_df['Aggregate rating'].mean()),
                'online_order_percentage': float(city_df['Has Online delivery'].apply(convert_yes_no).mean() * 100),
                'table_booking_percentage': float(city_df['Has Table booking'].apply(convert_yes_no).mean() * 100)
            }

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

            # Get mode safely
            city_mode = locality_df['City'].mode()
            price_mode = locality_df['Price range'].mode()

            stats = {
                'locality': locality,
                'city': city_mode[0] if len(city_mode) > 0 else 'Unknown',
                'count': int(len(locality_df)),
                'avg_rating': float(locality_df['Aggregate rating'].mean()),
                'price_range_mode': int(price_mode[0]) if len(price_mode) > 0 and pd.notna(price_mode[0]) else 2
            }
            locality_stats.append(stats)

        return locality_stats

    def get_insights(self):
        """Get interesting insights from location data"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return {}

        try:
            # Filter out Unknown cities for meaningful insights
            valid_cities = self.df[self.df['City'] != 'Unknown']
            
            if len(valid_cities) == 0:
                return {
                    'total_restaurants': len(self.df),
                    'total_cities': 0,
                    'total_localities': 0,
                    'highest_rated_city': 'N/A',
                    'highest_restaurant_density_city': 'N/A',
                    'avg_rating_overall': 0,
                    'most_expensive_city': 'N/A'
                }
            
            city_ratings = valid_cities.groupby('City')['Aggregate rating'].mean()
            city_counts = valid_cities['City'].value_counts()
            city_costs = valid_cities.groupby('City')['Average Cost for two'].mean()
            
            insights = {
                'total_restaurants': int(len(self.df)),
                'total_cities': int(self.df['City'].nunique()),
                'total_localities': int(self.df['Locality'].nunique()),
                'highest_rated_city': str(city_ratings.idxmax()) if len(city_ratings) > 0 else 'N/A',
                'highest_restaurant_density_city': str(city_counts.index[0]) if len(city_counts) > 0 else 'N/A',
                'avg_rating_overall': float(self.df['Aggregate rating'].mean()),
                'most_expensive_city': str(city_costs.idxmax()) if len(city_costs) > 0 else 'N/A'
            }
            return insights
        except Exception as e:
            print(f"Error calculating insights: {e}")
            return {
                'total_restaurants': len(self.df),
                'total_cities': 0,
                'total_localities': 0,
                'highest_rated_city': 'N/A',
                'highest_restaurant_density_city': 'N/A',
                'avg_rating_overall': 0,
                'most_expensive_city': 'N/A'
            }
        
    def get_map_data_optimized(self):
        """Get optimized location data with cuisine and cost for map visualization"""
        if self.df is None:
            self.load_data()
            if self.df is None:
                return []

        locations = []
        # Filter valid coordinates - return ALL restaurants with valid coordinates
        valid_df = self.df.dropna(subset=['Latitude', 'Longitude']).copy()
        
        # Sort by rating for better visualization (high-rated restaurants appear on top)
        valid_df = valid_df.sort_values('Aggregate rating', ascending=False)
        
        for idx, row in valid_df.iterrows():
            try:
                # Get first cuisine from comma-separated list
                cuisine = str(row.get('Cuisines', 'Unknown')).split(',')[0].strip() if pd.notna(row.get('Cuisines')) else 'Unknown'
                
                locations.append({
                    'lat': float(row['Latitude']),
                    'lng': float(row['Longitude']),
                    'name': str(row.get('Restaurant Name', 'Unknown')),
                    'city': str(row.get('City', 'Unknown')),
                    'rating': float(row.get('Aggregate rating', 0)),
                    'cuisine': cuisine,
                    'cost': float(row.get('Average Cost for two', 0)) if pd.notna(row.get('Average Cost for two')) else None
                })
            except (ValueError, TypeError):
                continue

        return locations

# Example usage (for testing)
if __name__ == "__main__":
    analyzer = LocationAnalyzer()
    analyzer.load_data()

    if analyzer.df is not None:
        print("Locations distribution (first 5):", analyzer.get_location_distribution()[:5])
        print("\nCity analysis:", analyzer.analyze_by_city())
        print("\nLocality analysis for New Delhi:", analyzer.analyze_by_locality('New Delhi'))
        print("\nGeneral insights:", analyzer.get_insights())
        print("\nMap data (first 5):", analyzer.get_map_data()[:5])