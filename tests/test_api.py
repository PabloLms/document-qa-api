import pytest
import asyncio
import pytest_asyncio
from httpx import AsyncClient
from app.main import app
from app.services.vector_store import clear_collection

@pytest_asyncio.fixture
async def client():
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture(autouse=True)
def clean_db():
    """Clear database before each test."""
    clear_collection()
    yield
    clear_collection()
@pytest.mark.asyncio
async def test_health_check(client):
    """Test health endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_ingest_document(client):
    """Test document ingestion."""
    payload = {
        "content": "FastAPI is a modern Python web framework. It is very fast and easy to use.",
        "metadata": {"source": "test"},
        "overlap": 20,
        "chunk_size":100
    }
    
    response = await client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_created"] > 0
    assert len(data["document_ids"]) > 0

@pytest.mark.asyncio
async def test_ingest_empty_document(client):
    """Test ingesting empty document."""
    payload = {"content": ""}
    
    response = await client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_ask_question(client):
    """Test asking a question."""
    # First ingest a document
    await client.post("/api/v1/ingest", json={
        "content": "FastAPI is a modern web framework for building APIs with Python.",
        "overlap": 5,
        "chunk_size":50,
        "metadata": {"source": "test"}
    })
    await asyncio.sleep(0.5)
    # Ask question
    response = await client.post("/api/v1/ask", json={
        "question": "What is FastAPI?",
        "top_k": 2
    })
    print(response)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0

@pytest.mark.asyncio
async def test_get_stats(client):
    """Test getting collection stats."""
    response = await client.get("/api/v1/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_documents" in data
    assert "collection_name" in data

@pytest.mark.asyncio
async def test_clear_collection(client):
    """Test clearing all documents."""
    # Add a document first
    await client.post("/api/v1/ingest", json={
        "content": "Test document"
    })
    
    # Clear
    response = await client.delete("/api/v1/clear")
    assert response.status_code == 200
    
    # Verify it's empty
    stats_response = await client.get("/api/v1/stats")
    stats = stats_response.json()
    assert stats["total_documents"] == 0