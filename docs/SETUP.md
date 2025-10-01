# Setup Guide

Detailed setup instructions for the Cognifyz ML Restaurant Analysis Platform.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Backend Setup](#backend-setup)
3. [Frontend Setup](#frontend-setup)
4. [Docker Setup](#docker-setup)
5. [Environment Configuration](#environment-configuration)
6. [Data Preparation](#data-preparation)
7. [Model Training](#model-training)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher
- **npm**: 9.0 or higher

### Optional
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+

## Backend Setup

### Step 1: Install Python

**Windows:**
Download from [python.org](https://www.python.org/downloads/) and run installer.

**macOS:**
```bash
brew install python@3.10
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### Step 2: Create Virtual Environment

```bash
cd backend
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import fastapi; import sklearn; import pandas; print('All packages installed successfully')"
```

### Step 6: Run Backend Server

```bash
python main.py
```

The server should start at `http://localhost:8000`

Access API documentation at `http://localhost:8000/docs`

## Frontend Setup

### Step 1: Install Node.js

**Windows/macOS:**
Download from [nodejs.org](https://nodejs.org/) and run installer.

**Linux:**
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Step 2: Verify Installation

```bash
node --version  # Should show v18.x.x or higher
npm --version   # Should show 9.x.x or higher
```

### Step 3: Install Dependencies

```bash
npm install
```

### Step 4: Run Development Server

```bash
npm run dev
```

The application should open at `http://localhost:5173`

### Step 5: Build for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

## Docker Setup

### Step 1: Install Docker

**Windows/macOS:**
Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)

**Linux:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### Step 2: Install Docker Compose

Docker Desktop includes Compose. For Linux:

```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Step 3: Build and Run

```bash
docker-compose up --build
```

Access the application:
- Frontend: `http://localhost`
- Backend: `http://localhost:8000`

### Step 4: Stop Services

```bash
docker-compose down
```

## Environment Configuration

Create a `.env` file in the project root (if needed for production):

```env
# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend Configuration
VITE_API_URL=http://localhost:8000

# Model Configuration
MODEL_PATH=backend/models
DATA_PATH=backend/data
```

## Data Preparation

### Step 1: Prepare Dataset

Ensure your dataset file is placed at:
```
backend/data/Dataset .csv
```

### Step 2: Verify Data Format

The dataset should include columns:
- Restaurant Name
- City
- Locality
- Cuisines
- Average Cost for two
- Price range
- Aggregate rating
- Votes
- Has Table booking
- Has Online delivery
- Latitude
- Longitude

### Step 3: Check Data

```bash
cd backend
python -c "import pandas as pd; df = pd.read_csv('data/Dataset .csv'); print(df.head()); print(df.info())"
```

## Model Training

### Initial Training

The models will train automatically when first used. For manual training:

```python
from tasks.rating_prediction import RatingPredictor
from tasks.cuisine_classification import CuisineClassifier

# Train rating prediction model
predictor = RatingPredictor()
results = predictor.train('data/Dataset .csv')
print(f"Rating Model R²: {results['r2']}")

# Train cuisine classification model
classifier = CuisineClassifier()
results = classifier.train('data/Dataset .csv')
print(f"Cuisine Model Accuracy: {results['accuracy']}")
```

### Verify Models

Check that model files exist:
```bash
ls -la backend/models/
```

You should see:
- rating_predictor.pkl
- rating_encoders.pkl
- cuisine_classifier.pkl
- cuisine_label_encoder.pkl

## Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError`
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Problem:** Port 8000 already in use
```bash
# Solution: Kill existing process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

**Problem:** CSV file not found
```bash
# Solution: Check file path
ls -la backend/data/
# Ensure file is named exactly "Dataset .csv" (with space)
```

### Frontend Issues

**Problem:** Build fails
```bash
# Solution: Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problem:** API connection fails
```bash
# Solution: Check backend is running
curl http://localhost:8000
# Update API URL in code if needed
```

### Docker Issues

**Problem:** Build fails
```bash
# Solution: Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

**Problem:** Port conflicts
```bash
# Solution: Stop conflicting services
docker-compose down
# Or modify ports in docker-compose.yml
```

### General Issues

**Problem:** Memory issues during training
```bash
# Solution: Reduce dataset size or increase system RAM
# Modify training code to use smaller batch sizes
```

**Problem:** Slow API responses
```bash
# Solution: Ensure models are pre-trained
# Check system resources (CPU, RAM)
```

## Next Steps

After successful setup:

1. Test all API endpoints using the Swagger UI at `http://localhost:8000/docs`
2. Explore the frontend interface at `http://localhost:5173`
3. Train models with your data
4. Customize the application for your needs

## Support

For additional help:
- Check the main README.md
- Review API documentation at `/docs`
- Contact the Cognifyz Technologies team

## Production Deployment

For production deployment:

1. Set up proper environment variables
2. Use production-grade database (if needed)
3. Configure proper CORS settings
4. Set up HTTPS with SSL certificates
5. Use production WSGI server (Gunicorn/Uvicorn workers)
6. Set up monitoring and logging
7. Configure automatic backups
8. Implement rate limiting and security measures
