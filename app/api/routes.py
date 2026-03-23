from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models import (
    DocumentInput, 
    QuestionInput, 
    AnswerResponse, 
    IngestResponse,
    HealthResponse
)
from app.services.embedding import process_document
from app.services.vector_store import (
    add_documents, 
    search_similar, 
    clear_collection,
    get_collection_stats
)
from app.services.llm import generate_answer
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(doc: DocumentInput):
    """
    Ingest a document into the vector store.
    Automatically chunks the document and creates embeddings.
    """
    try:
        # Process document (chunking)
        processed = process_document(doc.content)
        
        if not processed['chunks']:
            raise HTTPException(
                status_code=400, 
                detail="Document too short or empty after processing"
            )
        
        # Prepare metadata for each chunk
        metadatas = [
            {**(doc.metadata or {}), "chunk_index": i} 
            for i in range(len(processed['chunks']))
        ]
        
        # Add to vector store
        result = add_documents(
            texts=processed['chunks'],
            metadatas=metadatas
        )
        
        return IngestResponse(
            status="success",
            chunks_created=processed['total_chunks'],
            document_ids=result['ids'],
            message=f"Document split into {processed['total_chunks']} chunks"
        )
    
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(query: QuestionInput):
    """Ask a question (non-streaming response)."""
    try:
        # Search for relevant context
        similar_docs = search_similar(query.question, top_k=query.top_k)
        
        if not similar_docs:
            raise HTTPException(
                status_code=404,
                detail="No documents found in the database. Please ingest documents first."
            )
        
        # Build context from retrieved documents
        context = "\n\n---\n\n".join([
            f"Document {i+1}:\n{doc['content']}" 
            for i, doc in enumerate(similar_docs)
        ])
        
        # Generate answer
        answer = generate_answer(query.question, context, stream=False)
        
        return AnswerResponse(
            answer=answer,
            sources=similar_docs,
            model_used="claude-sonnet-4-20250514",
            chunks_used=len(similar_docs)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask-stream")
async def ask_question_stream(query: QuestionInput):
    """Ask a question with streaming response (Server-Sent Events)."""
    try:
        # Search for relevant context
        similar_docs = search_similar(query.question, top_k=query.top_k)
        
        if not similar_docs:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please ingest documents first."
            )
        
        # Build context
        context = "\n\n---\n\n".join([
            f"Document {i+1}:\n{doc['content']}" 
            for i, doc in enumerate(similar_docs)
        ])
        
        # Stream response
        async def event_stream():
            try:
                with generate_answer(query.question, context, stream=True) as stream:
                    for text in stream.text_stream:
                        yield f"data: {json.dumps({'text': text})}\n\n"
                
                # Send completion event
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            event_stream(), 
            media_type="text/event-stream"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
async def clear_all_documents():
    """Clear all documents from the vector store."""
    try:
        result = clear_collection()
        return result
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """Get collection statistics."""
    try:
        stats = get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))