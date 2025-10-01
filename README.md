# Cognifyz ML Restaurant Analysis Platform

A comprehensive full-stack machine learning application for restaurant data analysis, featuring rating prediction, personalized recommendations, cuisine classification, and location-based insights.

## Project Overview

This project is part of the Cognifyz Technologies Machine Learning Internship Program. It demonstrates practical applications of various ML techniques on real-world restaurant data through an intuitive web interface.

## Features

### 1. Restaurant Rating Prediction
- Predict restaurant ratings using regression models
- Based on features: votes, cost, price range, amenities
- Uses Random Forest Regression

### 2. Restaurant Recommendations
- Content-based filtering recommendation system
- Filter by cuisine, city, and price range
- Personalized suggestions based on user preferences

### 3. Cuisine Classification
- Multi-class classification of restaurant cuisines
- Trained on restaurant characteristics
- Provides confidence scores for predictions

### 4. Location-based Analysis
- Geospatial visualization and analysis
- City and locality statistics
- Restaurant distribution insights

## Technology Stack

### Backend
- **Framework**: Python FastAPI
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Model Persistence**: Joblib
- **API Documentation**: Swagger/OpenAPI

### Frontend
- **Framework**: React with TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Build Tool**: Vite

### DevOps
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **CI/CD**: GitHub Actions
- **Web Server**: Nginx

## Project Structure

```
project-root/
├── backend/
│   ├── data/              # Dataset files
│   ├── models/            # Saved ML models
│   ├── tasks/             # ML task modules
│   │   ├── rating_prediction.py
│   │   ├── recommendation_system.py
│   │   ├── cuisine_classification.py
│   │   └── location_analysis.py
│   ├── main.py            # FastAPI application
│   └── requirements.txt   # Python dependencies
├── frontend/
│   └── src/
│       ├── components/    # Reusable components
│       ├── pages/         # Page components
│       └── App.tsx        # Main application
├── .github/
│   └── workflows/
│       └── ci-cd.yml      # CI/CD pipeline
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # Service orchestration
└── README.md              # This file
```

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (optional, for containerized deployment)

### Local Development

#### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
python main.py
```

The backend API will be available at `http://localhost:8000`

#### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Docker Deployment

Build and run with Docker Compose:

```bash
docker-compose up --build
```

Access the application:
- Frontend: `http://localhost`
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## API Endpoints

### Rating Prediction
```bash
POST /api/predict-rating
Content-Type: application/json

{
  "votes": 100,
  "average_cost": 500,
  "price_range": 2,
  "has_table_booking": true,
  "has_online_delivery": false
}
```

### Restaurant Recommendations
```bash
POST /api/recommend-restaurants
Content-Type: application/json

{
  "cuisine": "Italian",
  "city": "New York",
  "price_range": 3,
  "top_n": 10
}
```

### Cuisine Classification
```bash
POST /api/classify-cuisine
Content-Type: application/json

{
  "aggregate_rating": 4.5,
  "votes": 200,
  "price_range": 3,
  "average_cost": 800,
  "has_table_booking": true,
  "has_online_delivery": true
}
```

### Location Analysis
```bash
GET /api/location-analysis
GET /api/location-analysis/map-data
GET /api/location-analysis/cities/{city}
```

## Testing API with cURL

### Predict Rating
```bash
curl -X POST "http://localhost:8000/api/predict-rating" \
  -H "Content-Type: application/json" \
  -d '{"votes": 150, "average_cost": 600, "price_range": 2, "has_table_booking": true, "has_online_delivery": false}'
```

### Get Recommendations
```bash
curl -X POST "http://localhost:8000/api/recommend-restaurants" \
  -H "Content-Type: application/json" \
  -d '{"cuisine": "Chinese", "city": "", "price_range": null, "top_n": 5}'
```

### Classify Cuisine
```bash
curl -X POST "http://localhost:8000/api/classify-cuisine" \
  -H "Content-Type: application/json" \
  -d '{"aggregate_rating": 4.2, "votes": 120, "price_range": 2, "average_cost": 500, "has_table_booking": false, "has_online_delivery": true}'
```

### Get Location Analysis
```bash
curl "http://localhost:8000/api/location-analysis"
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. **Backend Tests**: Installs Python dependencies and runs tests
2. **Frontend Build**: Type checks, lints, and builds the React application
3. **Docker Build**: Creates containerized images for deployment
4. **Deploy**: Deploys to staging/production environments

## ML Model Details

### Rating Prediction
- **Algorithm**: Random Forest Regressor
- **Features**: Votes, average cost, price range, amenities
- **Metrics**: MSE, R-squared
- **Model File**: `backend/models/rating_predictor.pkl`

### Recommendation System
- **Approach**: Content-based filtering
- **Features**: Cuisine, location, price range
- **Technique**: TF-IDF vectorization with cosine similarity

### Cuisine Classification
- **Algorithm**: Random Forest Classifier
- **Features**: Rating, votes, cost, amenities
- **Metrics**: Accuracy, precision, recall
- **Model File**: `backend/models/cuisine_classifier.pkl`

### Location Analysis
- **Techniques**: Geospatial analysis, statistical aggregation
- **Visualizations**: Distribution maps, city statistics

## Development Guidelines

### Adding New ML Tasks

1. Create a new module in `backend/tasks/`
2. Implement training and prediction methods
3. Add API endpoints in `backend/main.py`
4. Create frontend page in `src/pages/`
5. Update navigation in `src/App.tsx`

### Code Style

- Backend: Follow PEP 8 guidelines
- Frontend: Use ESLint configuration
- Use meaningful variable names
- Add docstrings for Python functions
- Type all TypeScript code

## Troubleshooting

### Backend Issues
- **Port already in use**: Change port in `main.py` or kill existing process
- **Module not found**: Ensure virtual environment is activated and dependencies installed
- **Data file not found**: Verify dataset is in `backend/data/` directory

### Frontend Issues
- **Build errors**: Run `npm run typecheck` to identify TypeScript errors
- **API connection failed**: Ensure backend is running on correct port
- **CORS errors**: Check CORS middleware configuration in backend

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is created as part of the Cognifyz Technologies internship program.

## Contact

For questions or support, please contact the Cognifyz Technologies team.

## Acknowledgments

- Cognifyz Technologies for the internship opportunity
- Dataset providers for restaurant data
- Open-source community for excellent tools and libraries
