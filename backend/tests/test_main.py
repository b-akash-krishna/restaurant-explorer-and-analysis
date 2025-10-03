import pytest
from httpx import AsyncClient
from main import app, fake_db, UserInDB
from auth.utils import get_password_hash, get_user_from_token
import time
import pytest
from fastapi.testclient import TestClient
from datetime import timedelta
from fastapi.responses import JSONResponse


# Use a fixture to clear the mock database before each test
@pytest.fixture(autouse=True)
def cleanup_fake_db():
    fake_db.clear()
    yield
    fake_db.clear()


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root endpoint for a successful response."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert "Cognifyz ML Restaurant Analysis API" in response.json()["message"]


@pytest.mark.asyncio
async def test_register_user_success():
    """Test successful user registration."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/auth/register", json={"username": "testuser", "password": "testpassword", "email": "test@example.com"})
    assert response.status_code == 200
    assert response.json()["message"] == "User registered successfully"
    assert "testuser" in fake_db
    assert isinstance(fake_db["testuser"], UserInDB)
    assert fake_db["testuser"].role == "user"


@pytest.mark.asyncio
async def test_register_user_conflict():
    """Test user registration with a username that already exists."""
    # First, register a user
    async with AsyncClient(app=app, base_url="http://test") as ac:
        await ac.post("/api/auth/register", json={"username": "testuser", "password": "testpassword", "email": "test@example.com"})
        # Try to register again with same username
        response = await ac.post("/api/auth/register", json={"username": "testuser", "password": "testpassword"})
    assert response.status_code == 409
    assert response.json()["detail"] == "Username already registered"


@pytest.mark.asyncio
async def test_login_for_access_token_success():
    """Test successful login and token generation."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Register a user first
        await ac.post("/api/auth/register", json={"username": "testuser", "password": "testpassword", "email": "test@example.com"})
        
        # Login
        response = await ac.post(
            "/api/auth/token",
            data={"username": "testuser", "password": "testpassword"}
        )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_for_access_token_failure():
    """Test login with incorrect password."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Register a user first
        await ac.post("/api/auth/register", json={"username": "testuser", "password": "testpassword", "email": "test@example.com"})
        
        # Try to login with wrong password
        response = await ac.post(
            "/api/auth/token",
            data={"username": "testuser", "password": "wrongpassword"}
        )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


@pytest.mark.asyncio
async def test_predict_rating_protected_endpoint_success(monkeypatch):
    """Test accessing a protected endpoint with a valid token."""
    
    # Mock the rating predictor to return a predictable value
    import main
    monkeypatch.setattr(main.rating_predictor, "predict", lambda features: 4.5)
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Register user
        await ac.post("/api/auth/register", json={"username": "testuser", "password": "testpassword", "email": "test@example.com"})
        
        # Login to get token
        token_response = await ac.post(
            "/api/auth/token",
            data={"username": "testuser", "password": "testpassword"}
        )
        token = token_response.json()["access_token"]
        
        # Access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = await ac.post(
            "/api/predict-rating",
            headers=headers,
            json={
                "votes": 10, "average_cost": 500, "price_range": 2,
                "has_table_booking": False, "has_online_delivery": True
            }
        )
    assert response.status_code == 200
    assert response.json()["success"] == True


@pytest.mark.asyncio
async def test_predict_rating_protected_endpoint_unauthorized():
    """Test accessing a protected endpoint without a token."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/predict-rating",
            json={
                "votes": 10, "average_cost": 500, "price_range": 2,
                "has_table_booking": False, "has_online_delivery": True
            }
        )
    # The response should be 401 and contain "Not authenticated" or "Could not validate credentials"
    assert response.status_code == 401
    # Check for either message since FastAPI's OAuth2PasswordBearer might return different messages
    detail = response.json().get("detail", "")
    assert "Not authenticated" in detail or "Could not validate credentials" in detail


@pytest.mark.asyncio
async def test_admin_endpoint_forbidden():
    """Test accessing the admin endpoint with a regular user token."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Register user
        await ac.post("/api/auth/register", json={"username": "testuser", "password": "testpassword", "email": "test@example.com"})
        
        # Login
        token_response = await ac.post(
            "/api/auth/token",
            data={"username": "testuser", "password": "testpassword"}
        )
        token = token_response.json()["access_token"]

        # Try to access admin endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = await ac.get("/api/admin/protected_data", headers=headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "You do not have permission to access this resource"


@pytest.mark.asyncio
async def test_admin_endpoint_success():
    """Test accessing the admin endpoint with an admin token."""
    # Manually create and store an admin user in the mock database
    fake_db["adminuser"] = UserInDB(
        username="adminuser",
        hashed_password=get_password_hash("adminpassword"),
        role="admin",
    )
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Login as admin
        token_response = await ac.post(
            "/api/auth/token",
            data={"username": "adminuser", "password": "adminpassword"}
        )
        token = token_response.json()["access_token"]

        # Access admin endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = await ac.get("/api/admin/protected_data", headers=headers)
    assert response.status_code == 200
    assert "Welcome, adminuser! This is a protected admin resource." in response.json()["message"]


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test if rate limiting is working on the /api/ping endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Send 10 requests, all should pass with the default limit of 10/minute
        for _ in range(10):
            response = await ac.get("/api/ping")
            assert response.status_code == 200

        # The 11th request should fail with a 429
        response = await ac.get("/api/ping")
    assert response.status_code == 429
    response_json = response.json()
    assert "detail" in response_json
    assert response_json["detail"] == "Too Many Requests"