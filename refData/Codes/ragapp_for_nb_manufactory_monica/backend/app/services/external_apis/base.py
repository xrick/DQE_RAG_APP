# backend/app/services/external_apis/base.py
from abc import ABC, abstractmethod
import aiohttp
import asyncio
from redis import Redis
from tenacity import retry, stop_after_attempt, wait_exponential

class BaseExternalAPI(ABC):
    def __init__(self):
        self.redis_client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT
        )
        self.session = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch(self, url: str, params: dict = None) -> dict:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        cache_key = f"{url}:{str(params)}"
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
            
        async with self.session.get(url, params=params) as response:
            result = await response.json()
            self.redis_client.setex(
                cache_key,
                300,  # 5 minutes cache
                json.dumps(result)
            )
            return result

# backend/app/services/external_apis/wikipedia.py
class WikipediaAPI(BaseExternalAPI):
    async def search(self, query: str) -> List[Dict[str, Any]]:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query
        }
        result = await self.fetch(settings.WIKIPEDIA_API_URL, params)
        return self._process_response(result)
