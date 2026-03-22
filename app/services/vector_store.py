import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(
    path="./chromadb_data",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

def add_documents(
    texts: List[str], 
    metadatas: Optional[List[Dict]] = None, 
    ids: Optional[List[str]] = None
) -> Dict:
    """
    Add documents to the vector store.
    
    Args:
        texts: List of text chunks
        metadatas: Optional metadata for each chunk
        ids: Optional IDs for each chunk (auto-generated if not provided)
    
    Returns:
        Dictionary with status and document IDs
    """
    if not texts:
        raise ValueError("No texts provided")
    
    # Generate IDs if not provided
    if ids is None:
        ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in range(len(texts))]
    
    # Add default metadata if not provided
    if metadatas is None:
        metadatas = [{}] * len(texts)
    
    # Add to ChromaDB (automatically generates embeddings)
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    return {
        "status": "success",
        "added": len(texts),
        "ids": ids
    }

def search_similar(query: str, top_k: int = 3) -> List[Dict]:
    """
    Search for similar documents using semantic search.
    
    Args:
        query: Search query
        top_k: Number of results to return
    
    Returns:
        List of similar documents with metadata
    """
    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count())  # Don't request more than available
    )
    
    # Format results
    similar_docs = []
    
    if results['ids'] and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            similar_docs.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            })
    
    return similar_docs

def get_collection_stats() -> Dict:
    """Get statistics about the collection."""
    count = collection.count()
    return {
        "total_documents": count,
        "collection_name": collection.name
    }

def clear_collection() -> Dict:
    """Clear all documents from the collection."""
    global collection
    
    # Delete and recreate collection
    chroma_client.delete_collection(name="documents")
    collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    return {"status": "cleared", "message": "All documents removed"}