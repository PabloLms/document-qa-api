import pytest
from app.services.embedding import chunk_text, process_document
from app.services.vector_store import add_documents, search_similar, clear_collection

def test_chunk_text_basic():
    """Test basic text chunking."""
    text = "a" * 2500  # 2500 characters
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    
    assert len(chunks) > 1, "Should create multiple chunks"
    assert all(len(chunk) <= 1000 for chunk in chunks), "Chunks should not exceed chunk_size"

def test_chunk_text_with_overlap():
    """Test that overlap is preserved."""
    text = "a" * 1500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    
    # With 200 char overlap, chunks should have common content
    if len(chunks) >= 2:
        # Last 200 chars of chunk 1 should match first 200 chars of chunk 2
        assert chunks[0][-200:] in chunks[1][:400], "Overlap should be preserved"

def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("")
    assert chunks == [], "Empty text should return empty list"

def test_process_document():
    """Test document processing."""
    content = "Test content. " * 100  # ~1400 chars
    result = process_document(content,200,20)
    
    assert "chunks" in result
    assert "total_chunks" in result
    assert "original_length" in result
    assert result["total_chunks"] > 0
    assert result["original_length"] == len(content)

def test_add_and_search_documents():
    """Test adding and searching documents in vector store."""    
    # Add test documents
    texts = [
        "Python is a programming language",
        "FastAPI is a web framework for Python",
        "Machine learning uses algorithms"
    ]
    
    result = add_documents(texts)
    assert result["status"] == "success"
    assert result["added"] == 3
    
    # Search
    results = search_similar("What is Python?", top_k=2)
    assert len(results) <= 2
    assert len(results) > 0
    
    # First result should be about Python
    assert "Python" in results[0]["content"] or "python" in results[0]["content"].lower()