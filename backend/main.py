import logging
from typing import List, Optional
from datetime import timedelta  # ADD THIS LINE

from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse  # ADD THIS LINE
from pydantic import BaseModel

from tasks.cuisine_classification import CuisineClassifier
from tasks.location_analysis import LocationAnalyzer
from tasks.rating_prediction import RatingPredictor
from tasks.recommendation_system import RestaurantRecommender

# --- SECURITY IMPORTS (Updated) ---
from auth.utils import (
    create_access_token,
    get_password_hash,
    get_user_from_token,
    verify_password,
)

# --- RATE LIMITING IMPORTS (New) ---
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from middleware.rate_limiter import limiter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- JWT Configuration (Updated) ---
# NOTE: The SECRET_KEY is now managed in auth/utils.py
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

# --- Database Mock (Updated) ---
# This is a temporary in-memory mock database.
fake_db = {}


# --- Pydantic Models (Updated for Roles) ---
class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str = "user"  # Default role for new users


class UserInDB(User):
    hashed_password: str


class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


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


# --- Main App (Updated with Rate Limiting) ---
app = FastAPI(title="Cognifyz ML Restaurant Analysis API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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

DATA_PATH = "data/Dataset.csv"

try:
    recommender.load_data(DATA_PATH)
    location_analyzer.load_data(DATA_PATH)
    logger.info("Data loaded successfully")
except Exception as e:
    logger.warning(f"Could not load data: {e}")


# --- Authentication Dependencies (Updated) ---
def get_current_user(token: str = Depends(oauth2_scheme)):
    user_data = get_user_from_token(token)
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username, role = user_data
    user = fake_db.get(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_current_active_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource",
        )
    return current_user


# --- Authentication Endpoints (Updated) ---
@app.post("/api/auth/register", tags=["auth"])
def register_user(user: UserRegister):
    """Register a new user."""
    if user.username in fake_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )
    hashed_password = get_password_hash(user.password)
    # Assign a default role of "user"
    user_in_db = UserInDB(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        role="user",
    )
    fake_db[user.username] = user_in_db
    return {"message": "User registered successfully"}


@app.post("/api/auth/token", tags=["auth"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get an access token."""
    user = fake_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# --- Endpoints (Updated) ---
@app.get("/")
def read_root():
    return {
        "message": "Cognifyz ML Restaurant Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "rating_prediction": "/api/predict-rating",
            "recommendations": "/api/recommend-restaurants",
            "cuisine_classification": "/api/classify-cuisine",
            "location_analysis": "/api/location-analysis",
            "authentication": "/api/auth/register",
        },
    }


@app.post("/api/predict-rating")
@limiter.limit("5/minute")
def predict_rating(
    request: Request,
    rating_request: RatingPredictionRequest, 
    current_user: User = Depends(get_current_user)
):
    """
    Predict restaurant rating based on features
    This endpoint now requires authentication.
    """
    try:
        features = [
            rating_request.votes,  # CHANGED from request.votes
            rating_request.average_cost,  # CHANGED from request.average_cost
            rating_request.price_range,  # CHANGED from request.price_range
            1 if rating_request.has_table_booking else 0,  # CHANGED from request.has_table_booking
            1 if rating_request.has_online_delivery else 0,  # CHANGED from request.has_online_delivery
        ]

        prediction = rating_predictor.predict(features)

        return {
            "success": True,
            "predicted_rating": round(prediction, 2),
            "features_used": {
                "votes": rating_request.votes,  # CHANGED from request.votes
                "average_cost": rating_request.average_cost,  # CHANGED from request.average_cost
                "price_range": rating_request.price_range,  # CHANGED from request.price_range
                "has_table_booking": rating_request.has_table_booking,  # CHANGED from request.has_table_booking
                "has_online_delivery": rating_request.has_online_delivery,  # CHANGED from request.has_online_delivery
            },
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
            top_n=request.top_n,
        )

        return {
            "success": True,
            "count": len(recommendations),
            "recommendations": recommendations,
            "filters_applied": {
                "cuisine": request.cuisine,
                "city": request.city,
                "price_range": request.price_range,
            },
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
            "cities": recommender.get_unique_cities(),
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
            1 if request.has_online_delivery else 0,
        ]

        result = cuisine_classifier.predict(features)

        return {
            "success": True,
            "predicted_cuisine": result["cuisine"],
            "confidence": round(result["confidence"] * 100, 2),
            "features_used": {
                "aggregate_rating": request.aggregate_rating,
                "votes": request.votes,
                "price_range": request.price_range,
                "average_cost": request.average_cost,
                "has_table_booking": request.has_table_booking,
                "has_online_delivery": request.has_online_delivery,
            },
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
            "locality_stats": location_analyzer.analyze_by_locality(),
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
            "locations": locations,
        }
    except Exception as e:
        logger.error(f"Map data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/location-analysis/cities/{city}")
def get_city_localities(city: str):
    """Get locality analysis for a specific city"""
    try:
        localities = location_analyzer.analyze_by_locality(city)

        return {"success": True, "city": city, "localities": localities}
    except Exception as e:
        logger.error(f"City localities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/protected_data")
def get_protected_data(current_user: User = Depends(get_current_active_admin_user)):
    """
    An example endpoint that is only accessible by an admin user.
    """
    return {"message": f"Welcome, {current_user.username}! This is a protected admin resource."}


@app.get("/api/ping")
@limiter.limit("10/minute")
def ping(request: Request):
    return {"message": "pong"}


@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too Many Requests"}
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

