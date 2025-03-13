# backend/app/services/rag_service.py
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import MultiQueryRetriever
from qdrant_client import QdrantClient
import asyncio
import aiohttp
from typing import List, Dict, Any

class RAGService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name="notebook_qa",
            embeddings=self.embeddings
        )
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),
            llm=ChatOpenAI(temperature=0)
        )
        
    async def get_internal_results(self, query: str) -> List[Dict[str, Any]]:
        docs = await self.retriever.aget_relevant_documents(query)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.similarity
            } for doc in docs
        ]

    async def get_external_results(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        tasks = [
            self._search_wikipedia(query),
            self._search_stackoverflow(query),
            self._search_reddit(query),
            self._search_duckduckgo(query)
        ]
        results = await asyncio.gather(*tasks)
        return {
            "wikipedia": results[0],
            "stackoverflow": results[1],
            "reddit": results[2],
            "duckduckgo": results[3]
        }
