# backend/app/api/endpoints/search.py
from fastapi import APIRouter, Depends
from typing import Dict, Any

router = APIRouter()

@router.post("/search")
async def search(
    query: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    internal_results = await rag_service.get_internal_results(query)
    external_results = await rag_service.get_external_results(query)
    
    return {
        "internal_results": internal_results,
        "external_results": external_results
    }
