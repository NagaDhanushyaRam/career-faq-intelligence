"""
FastAPI Backend for FAQ Recommendation System
Based on Listing 4 from the Midterm Report.

Endpoints:
- GET / : Health check
- POST /search : Search FAQs with query
- GET /stats : Get corpus statistics
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_TOP_K, DEFAULT_RETRIEVAL_MODE, API_HOST, API_PORT
from src.tfidf_retriever import TFIDFRetriever
from src.sbert_retriever import SBERTRetriever


# Global retrievers
tfidf_retriever: Optional[TFIDFRetriever] = None
sbert_retriever: Optional[SBERTRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global tfidf_retriever, sbert_retriever
    
    print("\n🚀 Starting FAQ Recommendation API...")
    
    # Initialize TF-IDF retriever
    print("📦 Loading TF-IDF model...")
    tfidf_retriever = TFIDFRetriever()
    try:
        tfidf_retriever.load()
        print("   ✅ TF-IDF model loaded from saved files")
    except FileNotFoundError:
        print("   ⚠️ No saved TF-IDF model found, building new index...")
        try:
            tfidf_retriever.build_index()
            tfidf_retriever.save()
        except FileNotFoundError as e:
            print(f"   ❌ Could not build TF-IDF index: {e}")
    
    # Initialize SBERT retriever
    print("📦 Loading SBERT model...")
    sbert_retriever = SBERTRetriever()
    try:
        sbert_retriever.load()
        print("   ✅ SBERT model loaded from saved files")
    except FileNotFoundError:
        print("   ⚠️ No saved SBERT model found, building new index...")
        try:
            sbert_retriever.build_index()
            sbert_retriever.save()
        except FileNotFoundError as e:
            print(f"   ❌ Could not build SBERT index: {e}")
    
    print("\n✅ API ready to serve requests!\n")
    
    yield
    
    # Cleanup on shutdown
    print("\n👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="FAQ Recommendation API",
    description="Dynamic FAQ Answer Recommendation System for Job & Careers Domain",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class SearchRequest(BaseModel):
    """Search request body."""
    query: str = Field(..., min_length=1, description="The search query")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="Number of results")
    mode: Literal["tfidf", "sbert"] = Field(
        default=DEFAULT_RETRIEVAL_MODE,
        description="Retrieval mode: 'tfidf' or 'sbert'"
    )
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")


class FAQResult(BaseModel):
    """Single FAQ result."""
    faq_id: int
    question: str
    answer: str
    score: float
    source: str
    retrieval_method: str


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    mode: str
    total_results: int
    results: List[FAQResult]


class StatsResponse(BaseModel):
    """Corpus statistics response."""
    total_faqs: int
    sources: dict
    tfidf_ready: bool
    sbert_ready: bool


# API Endpoints
@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "FAQ Recommendation API is running",
        "version": "1.0.0"
    }


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_faqs(request: SearchRequest):
    """
    Search for relevant FAQs based on user query.
    
    - **query**: The user's question or search query
    - **top_k**: Number of results to return (1-20)
    - **mode**: Retrieval method - 'tfidf' for keyword-based or 'sbert' for semantic
    - **threshold**: Minimum similarity score (0-1)
    """
    global tfidf_retriever, sbert_retriever
    
    # Select retriever based on mode
    if request.mode == "tfidf":
        if tfidf_retriever is None or not tfidf_retriever.is_fitted:
            raise HTTPException(
                status_code=503,
                detail="TF-IDF retriever not ready. Please run the data pipeline first."
            )
        retriever = tfidf_retriever
    else:
        if sbert_retriever is None or not sbert_retriever.is_fitted:
            raise HTTPException(
                status_code=503,
                detail="SBERT retriever not ready. Please run the data pipeline first."
            )
        retriever = sbert_retriever
    
    # Perform search
    try:
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        total_results=len(results),
        results=[FAQResult(**r) for r in results]
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_faqs_get(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=20),
    mode: Literal["tfidf", "sbert"] = Query(default=DEFAULT_RETRIEVAL_MODE),
    threshold: float = Query(default=0.0, ge=0.0, le=1.0)
):
    """Search FAQs using GET request (for easy testing)."""
    request = SearchRequest(
        query=query,
        top_k=top_k,
        mode=mode,
        threshold=threshold
    )
    return await search_faqs(request)


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
async def get_stats():
    """Get corpus statistics."""
    global tfidf_retriever, sbert_retriever
    
    stats = {
        "total_faqs": 0,
        "sources": {},
        "tfidf_ready": False,
        "sbert_ready": False
    }
    
    # Check TF-IDF
    if tfidf_retriever and tfidf_retriever.is_fitted:
        stats["tfidf_ready"] = True
        stats["total_faqs"] = len(tfidf_retriever.corpus_df)
        stats["sources"] = tfidf_retriever.corpus_df['source'].value_counts().to_dict()
    
    # Check SBERT
    if sbert_retriever and sbert_retriever.is_fitted:
        stats["sbert_ready"] = True
        if stats["total_faqs"] == 0:
            stats["total_faqs"] = len(sbert_retriever.corpus_df)
            stats["sources"] = sbert_retriever.corpus_df['source'].value_counts().to_dict()
    
    return StatsResponse(**stats)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

