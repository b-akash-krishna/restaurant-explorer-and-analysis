import os
from dotenv import load_dotenv

load_dotenv()

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from datetime import timedelta

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