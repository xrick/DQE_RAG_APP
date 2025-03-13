# backend/app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Notebook RAG System"
    API_V1_STR: str = "/api/v1"
    
    # Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # External APIs
    WIKIPEDIA_API_URL: str = "https://api.wikipedia.org/w/api.php"
    STACKOVERFLOW_API_URL: str = "https://api.stackexchange.com/2.3"
    REDDIT_API_URL: str = "https://oauth.reddit.com"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

settings = Settings()
