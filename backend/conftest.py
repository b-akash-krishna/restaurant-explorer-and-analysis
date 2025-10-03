import pytest
from httpx import AsyncClient
from main import app as fastapi_app

@pytest.fixture(scope="session")
async def client():
    async with AsyncClient(app=fastapi_app, base_url="http://test") as ac:
        yield ac