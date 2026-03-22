from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class LLMServiceError(Exception):
    """Custom exception for LLM service errors."""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_answer(question: str, context: str, stream: bool = False):
    """
    Generate answer using Claude with automatic retry on failures.
    
    Args:
        question: User's question
        context: Retrieved context from vector store
        stream: Whether to stream the response
    
    Returns:
        Streamed response or complete text
    """
    
    prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer concisely and accurately
- Use only information from the context
- If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided context."
- Do not make up information"""
    
    try:
        if stream:
            # Return the stream manager for streaming responses
            return client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
        else:
            # Return complete response
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    except Exception as e:
        raise LLMServiceError(f"Failed to generate answer: {str(e)}")

def test_api_key() -> bool:
    """Test if Anthropic API key is valid."""
    try:
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        return True
    except Exception:
        return False