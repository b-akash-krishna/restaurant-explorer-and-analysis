from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging

from tasks.rating_prediction import RatingPredictor
from tasks.recommendation_system import RestaurantRecommender
from tasks.cuisine_classification import CuisineClassifier
from tasks.location_analysis import LocationAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cognifyz ML Restaurant Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rating_predictor = RatingPredictor()
recommender = RestaurantRecommender()
cuisine_classifier = CuisineClassifier()
location_analyzer = LocationAnalyzer()

DATA_PATH = "backend/data/Dataset .csv"

try:
    recommender.load_data(DATA_PATH)
    location_analyzer.load_data(DATA_PATH)
    logger.info("Data loaded successfully")
except Exception as e:
    logger.warning(f"Could not load data: {e}")

class RatingPredictionRequest(BaseModel):
    votes: int
    average_cost: float
    price_range: int
    has_table_booking: bool
    has_online_delivery: bool

class RecommendationRequest(BaseModel):
    cuisine: Optional[str] = None
    city: Optional[str] = None
    price_range: Optional[int] = None
    top_n: int = 10

class CuisineClassificationRequest(BaseModel):
    aggregate_rating: float
    votes: int
    price_range: int
    average_cost: float
    has_table_booking: bool
    has_online_delivery: bool

@app.get("/")
def read_root():
    return {
        "message": "Cognifyz ML Restaurant Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "rating_prediction": "/api/predict-rating",
            "recommendations": "/api/recommend-restaurants",
            "cuisine_classification": "/api/classify-cuisine",
            "location_analysis": "/api/location-analysis"
        }
    }

@app.post("/api/predict-rating")
def predict_rating(request: RatingPredictionRequest):
    """Predict restaurant rating based on features"""
    try:
        features = [
            request.votes,
            request.average_cost,
            request.price_range,
            1 if request.has_table_booking else 0,
            1 if request.has_online_delivery else 0
        ]

        prediction = rating_predictor.predict(features)

        return {
            "success": True,
            "predicted_rating": round(prediction, 2),
            "features_used": {
                "votes": request.votes,
                "average_cost": request.average_cost,
                "price_range": request.price_range,
                "has_table_booking": request.has_table_booking,
                "has_online_delivery": request.has_online_delivery
            }
        }
    except Exception as e:
        logger.error(f"Rating prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend-restaurants")
def recommend_restaurants(request: RecommendationRequest):
    """Get restaurant recommendations based on user preferences"""
    try:
        recommendations = recommender.recommend(
            cuisine=request.cuisine,
            city=request.city,
            price_range=request.price_range,
            top_n=request.top_n
        )

        return {
            "success": True,
            "count": len(recommendations),
            "recommendations": recommendations,
            "filters_applied": {
                "cuisine": request.cuisine,
                "city": request.city,
                "price_range": request.price_range
            }
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend-restaurants/options")
def get_recommendation_options():
    """Get available cuisines and cities for filtering"""
    try:
        return {
            "success": True,
            "cuisines": recommender.get_unique_cuisines(),
            "cities": recommender.get_unique_cities()
        }
    except Exception as e:
        logger.error(f"Options error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify-cuisine")
def classify_cuisine(request: CuisineClassificationRequest):
    """Classify restaurant cuisine based on features"""
    try:
        features = [
            request.aggregate_rating,
            request.votes,
            request.price_range,
            request.average_cost,
            1 if request.has_table_booking else 0,
            1 if request.has_online_delivery else 0
        ]

        result = cuisine_classifier.predict(features)

        return {
            "success": True,
            "predicted_cuisine": result['cuisine'],
            "confidence": round(result['confidence'] * 100, 2),
            "features_used": {
                "aggregate_rating": request.aggregate_rating,
                "votes": request.votes,
                "price_range": request.price_range,
                "average_cost": request.average_cost,
                "has_table_booking": request.has_table_booking,
                "has_online_delivery": request.has_online_delivery
            }
        }
    except Exception as e:
        logger.error(f"Cuisine classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/location-analysis")
def get_location_analysis():
    """Get comprehensive location-based analysis"""
    try:
        return {
            "success": True,
            "insights": location_analyzer.get_insights(),
            "city_stats": location_analyzer.analyze_by_city(),
            "locality_stats": location_analyzer.analyze_by_locality()
        }
    except Exception as e:
        logger.error(f"Location analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/location-analysis/map-data")
def get_map_data():
    """Get location coordinates for map visualization"""
    try:
        locations = location_analyzer.get_location_distribution()

        return {
            "success": True,
            "count": len(locations),
            "locations": locations
        }
    except Exception as e:
        logger.error(f"Map data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/location-analysis/cities/{city}")
def get_city_localities(city: str):
    """Get locality analysis for a specific city"""
    try:
        localities = location_analyzer.analyze_by_locality(city)

        return {
            "success": True,
            "city": city,
            "localities": localities
        }
    except Exception as e:
        logger.error(f"City localities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
