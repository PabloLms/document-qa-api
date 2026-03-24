from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

class DocumentInput(BaseModel):
    """Input model for document ingestion."""
    content: str = Field(
        ..., 
        min_length=10, 
        max_length=100000,
        description="Document content to be indexed"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for the document"
    )
    chunk_size:int=Field(
        ...,
        ge=50,
        le=1000
    )
    overlap:int = Field(
        ...,
        ge=5,
        le=600
    )
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()

class QuestionInput(BaseModel):
    """Input model for questions."""
    question: str = Field(
        ..., 
        min_length=3, 
        max_length=1000,
        description="Question to ask about the documents"
    )
    top_k: int = Field(
        default=3, 
        ge=1, 
        le=10,
        description="Number of relevant chunks to retrieve"
    )
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class AnswerResponse(BaseModel):
    """Response model for answers."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents used for the answer"
    )
    model_used: str = Field(
        default="claude-sonnet-4-20250514",
        description="LLM model used"
    )
    chunks_used: int = Field(..., description="Number of chunks retrieved")

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    chunks_created: int
    document_ids: List[str]
    message: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"