from typing import List

def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        # Get chunk
        chunk = text[start:end].strip()
        
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - overlap
        
        # Prevent infinite loop
        if chunk_size <= overlap:
            break
    
    return chunks

def process_document(content: str) -> dict:
    """
    Process document: chunk and prepare metadata.
    
    Args:
        content: Document content
    
    Returns:
        Dictionary with chunks and metadata
    """
    chunks = chunk_text(content)
    
    return {
        "chunks": chunks,
        "total_chunks": len(chunks),
        "original_length": len(content),
        "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    }