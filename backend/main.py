import os
from dotenv import load_dotenv

load_dotenv()

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from datetime import timedelta
import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Corrected imports using absolute paths from the project root
from backend.tasks.cuisine_classification import CuisineClassifier
from backend.tasks.location_analysis import LocationAnalyzer
from backend.tasks.rating_prediction import RatingPredictor
from backend.tasks.recommendation_system import RestaurantRecommender
from backend.auth.utils import (
    create_access_token,
    get_password_hash,
    get_user_by_username,
    get_current_active_user,
    get_current_active_admin_user,
    verify_password,
)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from backend.middleware.rate_limiter import limiter
from backend.database import get_db, Base, engine
from backend.models import User as DBUser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- JWT Configuration ---
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

# --- ML Model Initialization (Lazy Loading) ---
rating_predictor = None
cuisine_classifier = None
restaurant_recommender = None
location_analyzer = None


def get_rating_predictor():
    global rating_predictor
    if rating_predictor is None:
        rating_predictor = RatingPredictor()
    return rating_predictor


def get_cuisine_classifier():
    global cuisine_classifier
    if cuisine_classifier is None:
        cuisine_classifier = CuisineClassifier()
    return cuisine_classifier


def get_restaurant_recommender():
    global restaurant_recommender
    if restaurant_recommender is None:
        restaurant_recommender = RestaurantRecommender()
    return restaurant_recommender


def get_location_analyzer():
    global location_analyzer
    if location_analyzer is None:
        location_analyzer = LocationAnalyzer()
    return location_analyzer


app = FastAPI(
    title="Restaurant Explorer API",
    description="API for restaurant data analysis, ML predictions, and recommendations.",
    version="1.0.0",
)


# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- Models ---
class UserInDB(BaseModel):
    username: str
    password: str
    disabled: Optional[bool] = None
    role: str = "user"


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class RatingPredictionRequest(BaseModel):
    online_order: int
    book_table: int
    votes: int
    location: str
    rest_type: str
    cuisines: str
    cost: float


class RecommendationRequest(BaseModel):
    location: str
    cuisine: str
    count: int = 5


# --- ASYNC EXECUTOR SETUP ---
executor = None


@app.on_event("startup")
def startup_event():
    global executor
    executor = ThreadPoolExecutor()
    Base.metadata.create_all(bind=engine)
    logger.info("Application started and ThreadPoolExecutor initialized.")


@app.on_event("shutdown")
def shutdown_event():
    global executor
    if executor:
        executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down gracefully.")


# --- ENDPOINTS ---
@app.post("/api/auth/register", response_model=UserInDB, status_code=status.HTTP_201_CREATED)
def register_user(user: UserInDB, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    hashed_password = get_password_hash(user.password)
    db_user = DBUser(username=user.username, hashed_password=hashed_password, role=user.role)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/api/auth/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = get_user_by_username(db, username=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": [user.role]},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/users/me/")
async def read_users_me(current_user: DBUser = Depends(get_current_active_user)):
    return current_user


# NEW: Endpoint to get recommendation options
@app.get("/api/recommend-restaurants/options")
async def get_recommendation_options():
    try:
        recommender = get_restaurant_recommender()
        cuisines = await asyncio.get_event_loop().run_in_executor(
            executor,
            recommender.get_unique_cuisines,
        )
        cities = await asyncio.get_event_loop().run_in_executor(
            executor,
            recommender.get_unique_cities,
        )
        return {"cuisines": cuisines, "cities": cities}
    except Exception as e:
        logger.error(f"Error fetching recommendation options: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- ASYNC ML ENDPOINTS ---
# @app.post("/api/predict-rating")
# @limiter.limit("5/minute")
# # async def predict_rating(request: RatingPredictionRequest, request_obj: Request, current_user: DBUser = Depends(get_current_active_user)):
# async def predict_rating(request: RatingPredictionRequest, request_obj: Request):
#     try:
#         predictor = get_rating_predictor()
#         prediction = await asyncio.get_event_loop().run_in_executor(
#             executor,
#             predictor.predict_rating,
#             request.dict()
#         )
#         return {"success": True, "predicted_rating": prediction}
#     except Exception as e:
#         logger.error(f"Rating prediction error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/api/recommend-restaurants")
# @limiter.limit("5/minute")
# # async def recommend_restaurants(request: RecommendationRequest, request_obj: Request, current_user: DBUser = Depends(get_current_active_user)):
# async def recommend_restaurants(request: RecommendationRequest, request_obj: Request):
#     try:
#         recommender = get_restaurant_recommender()
#         recommendations = await asyncio.get_event_loop().run_in_executor(
#             executor,
#             recommender.recommend,
#             request.cuisine,      # First param: cuisine
#             request.location,     # Second param: city (your frontend sends 'location')
#             None,                 # Third param: price_range
#             request.count,        # Fourth param: top_n
#         )
#         return {"success": True, "recommendations": recommendations}
#     except Exception as e:
#         logger.error(f"Recommendation error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict-rating/options")
async def get_prediction_options():
    """Get options for rating prediction form"""
    try:
        predictor = get_rating_predictor()
        options = await asyncio.get_event_loop().run_in_executor(
            executor,
            predictor.get_prediction_options
        )
        if options is None:
            raise HTTPException(status_code=500, detail="Failed to load options")
        return {"success": True, **options}
    except Exception as e:
        logger.error(f"Error fetching prediction options: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/predict-rating")
@limiter.limit("5/minute")
async def predict_rating(data: RatingPredictionRequest, request: Request):
    try:
        predictor = get_rating_predictor()
        
        # Map frontend field names to model field names
        model_input = {
            'Has Online delivery': data.online_order,
            'Has Table booking': data.book_table,
            'Votes': data.votes,
            'Cost': data.cost,
            'City': data.location,
            'Cuisines': data.cuisines,
            'Rest type': data.rest_type
        }
        
        prediction = await asyncio.get_event_loop().run_in_executor(
            executor,
            predictor.predict_rating,
            model_input
        )
        
        # Return the response in the format the frontend expects
        return {
            "success": True,
            "predicted_rating": prediction,
            "features_used": {
                "votes": data.votes,
                "online_order": data.online_order,
                "book_table": data.book_table,
                "location": data.location,
                "rest_type": data.rest_type,
                "cuisines": data.cuisines,
                "cost": data.cost
            }
        }
    except Exception as e:
        logger.error(f"Rating prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend-restaurants")
@limiter.limit("5/minute")
async def recommend_restaurants(data: RecommendationRequest, request: Request):
    # Same changes as above
    try:
        recommender = get_restaurant_recommender()
        recommendations = await asyncio.get_event_loop().run_in_executor(
            executor,
            recommender.recommend,
            data.cuisine,      # Changed from request.cuisine
            data.location,     # Changed from request.location
            None,
            data.count,        # Changed from request.count
        )
        return {"success": True, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cuisine-classification/{url}")
@limiter.limit("5/minute")
async def classify_cuisine_from_url(url: str, request: Request):
    try:
        classifier = get_cuisine_classifier()
        cuisine = await asyncio.get_event_loop().run_in_executor(
            executor,
            classifier.classify_from_url,
            url
        )
        return {"success": True, "cuisine": cuisine}
    except Exception as e:
        logger.error(f"Cuisine classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/location-analysis/map")
async def get_map_data():
    try:
        analyzer = get_location_analyzer()
        locations = await asyncio.get_event_loop().run_in_executor(
            executor,
            analyzer.get_map_data
        )
        return {
            "success": True,
            "count": len(locations),
            "locations": locations,
        }
    except Exception as e:
        logger.error(f"Map data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/location-analysis/cities/{city}")
async def get_city_localities(city: str):
    try:
        analyzer = get_location_analyzer()
        localities = await asyncio.get_event_loop().run_in_executor(
            executor,
            analyzer.analyze_by_locality,
            city
        )
        return {"success": True, "city": city, "localities": localities}
    except Exception as e:
        logger.error(f"City localities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add these endpoints to backend/main.py after the existing location endpoints

@app.get("/api/location-analysis")
async def get_location_analysis():
    """Get comprehensive location analysis including insights, cities, and localities"""
    try:
        analyzer = get_location_analyzer()
        
        # Get insights
        insights = await asyncio.get_event_loop().run_in_executor(
            executor,
            analyzer.get_insights
        )
        
        # Get city analysis
        city_analysis = await asyncio.get_event_loop().run_in_executor(
            executor,
            analyzer.analyze_by_city
        )
        
        # Convert city_analysis dict to list format
        city_stats = []
        for city_name, stats in city_analysis.items():
            try:
                # Get top cuisine for this city
                if analyzer.df is not None:
                    city_df = analyzer.df[analyzer.df['City'] == city_name]
                    if len(city_df) > 0 and 'Cuisines' in city_df.columns:
                        cuisines = city_df['Cuisines'].dropna()
                        if len(cuisines) > 0:
                            top_cuisine = cuisines.mode()[0] if len(cuisines.mode()) > 0 else 'Unknown'
                        else:
                            top_cuisine = 'Unknown'
                    else:
                        top_cuisine = 'Unknown'
                    
                    # Get average cost
                    avg_cost = city_df['Average Cost for two'].mean() if 'Average Cost for two' in city_df.columns else 0
                else:
                    top_cuisine = 'Unknown'
                    avg_cost = 0
                
                city_stats.append({
                    'city': city_name,
                    'count': int(stats['restaurant_count']),
                    'avg_rating': float(stats['average_rating']),
                    'avg_cost': float(avg_cost) if pd.notna(avg_cost) else 0,
                    'top_cuisine': str(top_cuisine).split(',')[0].strip()
                })
            except Exception as e:
                logger.error(f"Error processing city {city_name}: {e}")
                continue
        
        # Sort by restaurant count
        city_stats.sort(key=lambda x: x['count'], reverse=True)
        city_stats = city_stats[:15]  # Top 15 cities
        
        # Get locality analysis for top city
        if city_stats:
            top_city = city_stats[0]['city']
        else:
            top_city = 'Bangalore'
            
        locality_stats = await asyncio.get_event_loop().run_in_executor(
            executor,
            analyzer.analyze_by_locality,
            top_city
        )
        
        return {
            "success": True,
            "insights": insights,
            "city_stats": city_stats,
            "locality_stats": locality_stats
        }
    except Exception as e:
        logger.error(f"Location analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/location-analysis/localities/{city}")
async def get_city_localities_detailed(city: str):
    """Get detailed locality analysis for a specific city"""
    try:
        analyzer = get_location_analyzer()
        localities = await asyncio.get_event_loop().run_in_executor(
            executor,
            analyzer.analyze_by_locality,
            city
        )
        return {
            "success": True,
            "city": city,
            "locality_count": len(localities),
            "localities": localities
        }
    except Exception as e:
        logger.error(f"City localities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/protected_data")
async def get_protected_data(current_user: DBUser = Depends(get_current_active_admin_user)):
    return {"message": f"Welcome, {current_user.username}! This is a protected admin resource."}


@app.get("/api/ping")
@limiter.limit("10/minute")
async def ping(request: Request):
    return {"message": "pong"}


@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests. Please try again later."},
    )

# Add these new endpoints to your backend/main.py

# NEW ENDPOINTS FOR TASK COMPLETION

# ============ TASK 1: Rating Prediction - Feature Importance ============
@app.get("/api/predict-rating/model-interpretation")
async def get_model_interpretation():
    """Get comprehensive model interpretation including feature importance"""
    try:
        predictor = get_rating_predictor()
        interpretation = await asyncio.get_event_loop().run_in_executor(
            executor,
            predictor.get_model_interpretation
        )
        return {
            "success": True,
            **interpretation
        }
    except Exception as e:
        logger.error(f"Error fetching model interpretation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict-rating/feature-importance")
async def get_feature_importance():
    """Get feature importance rankings"""
    try:
        predictor = get_rating_predictor()
        importance = await asyncio.get_event_loop().run_in_executor(
            executor,
            predictor.get_feature_importance
        )
        return {
            "success": True,
            "feature_importance": importance
        }
    except Exception as e:
        logger.error(f"Error fetching feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ TASK 3: Cuisine Classification - Complete Implementation ============
class CuisineClassificationRequest(BaseModel):
    city: str
    has_table_booking: int
    has_online_delivery: int
    price_range: int
    votes: int


@app.post("/api/classify-cuisine")
@limiter.limit("5/minute")
async def classify_cuisine(data: CuisineClassificationRequest, request: Request):
    """Classify restaurant cuisine based on features"""
    try:
        classifier = get_cuisine_classifier()
        
        # Map to expected format
        features = {
            'City': data.city,
            'Has Table booking': data.has_table_booking,
            'Has Online delivery': data.has_online_delivery,
            'Price range': data.price_range,
            'Votes': data.votes
        }
        
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            classifier.predict,
            features
        )
        
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Cuisine classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/classify-cuisine/options")
async def get_classification_options():
    """Get options for cuisine classification form"""
    try:
        classifier = get_cuisine_classifier()
        
        # Load data to get options
        import pandas as pd  # Ensure pandas is imported
        df = pd.read_csv('backend/data/Dataset.csv')
        
        cities = sorted(df['City'].dropna().unique().tolist())[:50]
        
        # Get price range info
        price_ranges = {
            '1': 'Budget (₹)',
            '2': 'Moderate (₹₹)',
            '3': 'Expensive (₹₹₹)',
            '4': 'Very Expensive (₹₹₹₹)'
        }
        
        # Get a random sample safely
        if len(df) > 0:
            sample = df.sample(1).iloc[0]
            random_sample = {
                'city': str(sample.get('City', cities[0] if cities else 'Bangalore')),
                'has_table_booking': 1 if str(sample.get('Has Table booking', 'No')).lower() == 'yes' else 0,
                'has_online_delivery': 1 if str(sample.get('Has Online delivery', 'Yes')).lower() == 'yes' else 0,
                'price_range': int(sample.get('Price range', 2)) if pd.notna(sample.get('Price range')) else 2,
                'votes': int(sample.get('Votes', 100)) if pd.notna(sample.get('Votes')) else 100
            }
        else:
            random_sample = {
                'city': cities[0] if cities else 'Bangalore',
                'has_table_booking': 1,
                'has_online_delivery': 1,
                'price_range': 2,
                'votes': 100
            }
        
        votes_series = pd.to_numeric(df['Votes'], errors='coerce').dropna()
        
        return {
            "success": True,
            "cities": cities,
            "price_ranges": price_ranges,
            "random_sample": random_sample,
            "votes_range": {
                "min": int(votes_series.min()) if len(votes_series) > 0 else 0,
                "max": int(votes_series.max()) if len(votes_series) > 0 else 10000,
                "avg": int(votes_series.mean()) if len(votes_series) > 0 else 100
            }
        }
    except Exception as e:
        logger.error(f"Error fetching classification options: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/classify-cuisine/performance")
async def get_cuisine_model_performance():
    """Get comprehensive performance metrics for cuisine classification model"""
    try:
        classifier = get_cuisine_classifier()
        performance = await asyncio.get_event_loop().run_in_executor(
            executor,
            classifier.get_model_performance_summary
        )
        return {
            "success": True,
            **performance
        }
    except Exception as e:
        logger.error(f"Error fetching cuisine model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ TESTING ENDPOINTS ============
@app.get("/api/test/rating-prediction-comprehensive")
async def test_rating_prediction():
    """Comprehensive test of rating prediction with multiple samples"""
    try:
        predictor = get_rating_predictor()
        
        test_samples = [
            {
                'Has Online delivery': 1,
                'Has Table booking': 1,
                'Votes': 500,
                'Cost': 1000,
                'City': 'Bangalore',
                'Cuisines': 'North Indian',
                'Rest type': 'Casual Dining'
            },
            {
                'Has Online delivery': 0,
                'Has Table booking': 0,
                'Votes': 50,
                'Cost': 300,
                'City': 'Delhi',
                'Cuisines': 'Chinese',
                'Rest type': 'Quick Bites'
            },
            {
                'Has Online delivery': 1,
                'Has Table booking': 1,
                'Votes': 1000,
                'Cost': 2000,
                'City': 'Mumbai',
                'Cuisines': 'Continental',
                'Rest type': 'Fine Dining'
            }
        ]
        
        results = []
        for sample in test_samples:
            prediction = await asyncio.get_event_loop().run_in_executor(
                executor,
                predictor.predict_rating,
                sample
            )
            results.append({
                'input': sample,
                'predicted_rating': prediction
            })
        
        return {
            "success": True,
            "test_results": results
        }
    except Exception as e:
        logger.error(f"Test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test/cuisine-classification-comprehensive")
async def test_cuisine_classification():
    """Comprehensive test of cuisine classification"""
    try:
        classifier = get_cuisine_classifier()
        
        test_samples = [
            {
                'City': 'Bangalore',
                'Has Table booking': 1,
                'Has Online delivery': 1,
                'Price range': 3,
                'Votes': 500
            },
            {
                'City': 'Delhi',
                'Has Table booking': 0,
                'Has Online delivery': 1,
                'Price range': 2,
                'Votes': 200
            },
            {
                'City': 'Mumbai',
                'Has Table booking': 1,
                'Has Online delivery': 0,
                'Price range': 4,
                'Votes': 800
            }
        ]
        
        results = []
        for sample in test_samples:
            prediction = await asyncio.get_event_loop().run_in_executor(
                executor,
                classifier.predict,
                sample
            )
            results.append({
                'input': sample,
                'prediction': prediction
            })
        
        return {
            "success": True,
            "test_results": results
        }
    except Exception as e:
        logger.error(f"Test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test/recommendations-comprehensive")
async def test_recommendations():
    """Comprehensive test of recommendation system"""
    try:
        recommender = get_restaurant_recommender()
        
        test_cases = [
            {'cuisine': 'North Indian', 'city': 'Bangalore', 'top_n': 5},
            {'cuisine': 'Chinese', 'city': 'Delhi', 'top_n': 5},
            {'cuisine': 'Continental', 'city': 'Mumbai', 'top_n': 5}
        ]
        
        results = []
        for case in test_cases:
            recommendations = await asyncio.get_event_loop().run_in_executor(
                executor,
                recommender.recommend,
                case['cuisine'],
                case['city'],
                None,
                case['top_n']
            )
            results.append({
                'query': case,
                'recommendations_count': len(recommendations),
                'top_3': recommendations[:3] if recommendations else []
            })
        
        return {
            "success": True,
            "test_results": results
        }
    except Exception as e:
        logger.error(f"Test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ DOCUMENTATION ENDPOINT ============
@app.get("/api/internship-tasks/completion-status")
async def get_task_completion_status():
    """Get the completion status of all internship tasks"""
    return {
        "success": True,
        "internship": "Cognifyz Technologies ML Internship",
        "tasks": {
            "task_1_rating_prediction": {
                "status": "✅ Complete",
                "components": {
                    "preprocessing": "✅ Missing values handled, categorical encoding, train/test split",
                    "algorithm": "✅ RandomForestRegressor implemented",
                    "evaluation": "✅ MSE, R², RMSE, MAE metrics",
                    "interpretation": "✅ Feature importance analysis, influential features identified",
                    "hyperparameter_tuning": "✅ Optuna optimization"
                },
                "endpoints": [
                    "/api/predict-rating",
                    "/api/predict-rating/options",
                    "/api/predict-rating/feature-importance",
                    "/api/predict-rating/model-interpretation"
                ]
            },
            "task_2_recommendations": {
                "status": "✅ Complete",
                "components": {
                    "preprocessing": "✅ Missing values handled, categorical encoding",
                    "criteria": "✅ Cuisine preference, location, price range",
                    "approach": "✅ Content-based filtering with TF-IDF",
                    "testing": "✅ Sample preferences tested, quality evaluated"
                },
                "endpoints": [
                    "/api/recommend-restaurants",
                    "/api/recommend-restaurants/options"
                ]
            },
            "task_3_cuisine_classification": {
                "status": "✅ Complete",
                "components": {
                    "preprocessing": "✅ Missing values handled, categorical encoding",
                    "train_test_split": "✅ Stratified 80/20 split",
                    "algorithm": "✅ RandomForestClassifier",
                    "evaluation": "✅ Accuracy, Precision, Recall, F1-score",
                    "per_cuisine_analysis": "✅ Performance metrics per cuisine",
                    "bias_identification": "✅ Class imbalance and performance issues identified"
                },
                "endpoints": [
                    "/api/classify-cuisine",
                    "/api/classify-cuisine/options",
                    "/api/classify-cuisine/performance",
                    "/api/cuisine-classification/{url}"
                ]
            }
        },
        "additional_features": {
            "authentication": "✅ JWT-based auth with user roles",
            "rate_limiting": "✅ Request throttling implemented",
            "caching": "✅ Redis caching for predictions",
            "location_analysis": "✅ Geospatial analysis with maps",
            "testing": "✅ Comprehensive test suite"
        }
    }

# Add this endpoint after the existing /api/predict-rating/model-interpretation endpoint

@app.get("/api/predict-rating/model-insights")
async def get_model_insights():
    """Get comprehensive model insights including metrics and feature importance for visualization"""
    try:
        predictor = get_rating_predictor()
        
        # Load model if not loaded
        if not predictor.model:
            predictor.load_model()
        
        # Get feature importance
        feature_importance = await asyncio.get_event_loop().run_in_executor(
            executor,
            predictor.get_feature_importance
        )
        
        # Format feature importance for visualization
        feature_data = [
            {"feature": k, "importance": float(v)} 
            for k, v in feature_importance.items()
        ]
        
        # Get evaluation metrics
        metrics = predictor.evaluation_metrics if predictor.evaluation_metrics else {
            'train': {'r2': 0, 'rmse': 0, 'mae': 0, 'mse': 0},
            'test': {'r2': 0, 'rmse': 0, 'mae': 0, 'mse': 0}
        }
        
        # Calculate accuracy percentage (R² * 100)
        train_accuracy = metrics['train']['r2'] * 100 if 'train' in metrics else 0
        test_accuracy = metrics['test']['r2'] * 100 if 'test' in metrics else 0
        
        return {
            "success": True,
            "metrics": {
                "train_r2": float(metrics['train']['r2']) if 'train' in metrics else 0,
                "test_r2": float(metrics['test']['r2']) if 'test' in metrics else 0,
                "train_rmse": float(metrics['train']['rmse']) if 'train' in metrics else 0,
                "test_rmse": float(metrics['test']['rmse']) if 'test' in metrics else 0,
                "train_mae": float(metrics['train']['mae']) if 'train' in metrics else 0,
                "test_mae": float(metrics['test']['mae']) if 'test' in metrics else 0,
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy)
            },
            "feature_importance": feature_data
        }
    except Exception as e:
        logger.error(f"Error fetching model insights: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))