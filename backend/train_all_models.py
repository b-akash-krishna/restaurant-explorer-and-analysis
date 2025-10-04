#!/usr/bin/env python3
"""
Comprehensive Training Script for All ML Models
Cognifyz Technologies ML Internship

This script trains all three models required for the internship tasks:
1. Rating Prediction (Task 1)
2. Cuisine Classification (Task 3)
3. Recommendation System (Task 2)
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from tasks.rating_prediction import RatingPredictor
from tasks.cuisine_classification import CuisineClassifier
from tasks.recommendation_system import RestaurantRecommender

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def train_rating_predictor():
    """Train the rating prediction model (Task 1)"""
    print_header("TASK 1: Training Rating Prediction Model")
    
    predictor = RatingPredictor()
    data_path = os.path.join(backend_dir, 'data', 'Dataset.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        return False
    
    print("📊 Starting model training with hyperparameter tuning...")
    print("This may take several minutes...\n")
    
    try:
        # Train with hyperparameter tuning
        results = predictor.tune_hyperparameters(n_trials=20, data_path=data_path)
        
        if 'error' in results:
            print(f"❌ Training failed: {results['error']}")
            return False
        
        print("\n✅ Rating Prediction Model Training Complete!")
        print("\nFinal Model Performance:")
        print(f"  Training R²: {results['train_metrics']['r2']:.4f}")
        print(f"  Test R²: {results['test_metrics']['r2']:.4f}")
        print(f"  Test RMSE: {results['test_metrics']['rmse']:.4f}")
        print(f"  Test MAE: {results['test_metrics']['mae']:.4f}")
        
        print("\nTop 5 Most Influential Features:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5], 1):
            print(f"  {i}. {feature}: {importance:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_cuisine_classifier():
    """Train the cuisine classification model (Task 3)"""
    print_header("TASK 3: Training Cuisine Classification Model")
    
    classifier = CuisineClassifier()
    data_path = os.path.join(backend_dir, 'data', 'Dataset.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        return False
    
    print("📊 Starting model training...")
    print("This may take a few minutes...\n")
    
    try:
        results = classifier.train(data_path=data_path)
        
        if 'error' in results:
            print(f"❌ Training failed: {results['error']}")
            return False
        
        print("\n✅ Cuisine Classification Model Training Complete!")
        print("\nModel Performance:")
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"  Test Precision (Macro): {results['test_metrics']['precision_macro']:.4f}")
        print(f"  Test Recall (Macro): {results['test_metrics']['recall_macro']:.4f}")
        print(f"  Test F1-Score (Macro): {results['test_metrics']['f1_macro']:.4f}")
        
        print("\nPer-Cuisine Performance (Top 5):")
        sorted_cuisines = sorted(
            results['per_cuisine_metrics'].items(),
            key=lambda x: x[1]['f1-score'],
            reverse=True
        )
        for cuisine, metrics in sorted_cuisines[:5]:
            print(f"  {cuisine}: F1={metrics['f1-score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        
        if results['biases_identified']:
            print("\n⚠️  Note: Some biases or performance issues were identified during training.")
            print("   Check the detailed output above for more information.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_recommendation_system():
    """Setup the recommendation system (Task 2)"""
    print_header("TASK 2: Setting Up Recommendation System")
    
    recommender = RestaurantRecommender()
    
    print("📊 Loading and preprocessing data for recommendations...")
    
    try:
        recommender.load_data()
        
        if recommender.df is None:
            print("❌ Failed to load data")
            return False
        
        print(f"\n✅ Recommendation System Ready!")
        print(f"\nDataset Statistics:")
        print(f"  Total Restaurants: {len(recommender.df)}")
        print(f"  Unique Cities: {recommender.df['City'].nunique()}")
        print(f"  Unique Cuisines: {len(recommender.get_unique_cuisines())}")
        
        # Save the processed data
        recommender.save_model()
        print("\n✅ Recommendation data saved successfully!")
        
        # Test the system
        print("\n🧪 Testing recommendation system...")
        test_recommendations = recommender.recommend(
            cuisine='North Indian',
            city='Bangalore',
            top_n=5
        )
        
        if test_recommendations:
            print(f"✅ Test passed! Found {len(test_recommendations)} recommendations")
            print("\nSample Recommendation:")
            if test_recommendations:
                rec = test_recommendations[0]
                print(f"  {rec['name']}")
                print(f"  Cuisine: {rec['cuisine']}")
                print(f"  City: {rec['city']}")
                print(f"  Rating: {rec['rating']}")
        else:
            print("⚠️  Test returned no recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    print_header("Cognifyz Technologies ML Internship - Model Training")
    print("This script will train all models required for the internship tasks.")
    print("\nTasks to be completed:")
    print("  1. Rating Prediction (Regression)")
    print("  2. Restaurant Recommendation (Content-Based Filtering)")
    print("  3. Cuisine Classification (Multi-Class Classification)")
    
    input("\nPress Enter to start training...")
    
    results = {
        'rating_prediction': False,
        'cuisine_classification': False,
        'recommendation_system': False
    }
    
    # Train all models
    results['rating_prediction'] = train_rating_predictor()
    results['cuisine_classification'] = train_cuisine_classifier()
    results['recommendation_system'] = setup_recommendation_system()
    
    # Print summary
    print_header("Training Summary")
    
    all_success = all(results.values())
    
    print("Task Completion Status:")
    print(f"  Task 1 - Rating Prediction: {'✅ Success' if results['rating_prediction'] else '❌ Failed'}")
    print(f"  Task 2 - Recommendations: {'✅ Success' if results['recommendation_system'] else '❌ Failed'}")
    print(f"  Task 3 - Cuisine Classification: {'✅ Success' if results['cuisine_classification'] else '❌ Failed'}")
    
    if all_success:
        print("\n🎉 All models trained successfully!")
        print("\nYou can now:")
        print("  1. Start the FastAPI backend: uvicorn backend.main:app --reload")
        print("  2. Start the frontend: npm run dev")
        print("  3. Test the models through the web interface")
        print("\nAPI Documentation: http://localhost:8000/docs")
    else:
        print("\n⚠️  Some models failed to train. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())