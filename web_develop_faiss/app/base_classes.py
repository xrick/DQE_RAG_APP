from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

class BasePromptTemplate(ABC):
    @abstractmethod
    def format(self, **kwargs) -> str:
        pass

class ModelManager:
    def __init__(self):
        self.models = {
            "gpt-4": "OpenAI GPT-4",
            "gpt-3.5-turbo": "OpenAI GPT-3.5",
            "assistant-default": "OpenAI Assistant Default",
            "assistant-custom": "OpenAI Assistant Custom"
        }
        
    def get_available_models(self) -> Dict[str, str]:
        return self.models
    
    def validate_model(self, model_id: str) -> bool:
        return model_id in self.models

class BaseLLMService(ABC):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_manager = ModelManager()
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs):
        pass

class BaseServiceFlow(ABC):
    pass
# class AssistantPromptTemplate(BasePromptTemplate):
#     def format(self, **kwargs) -> str:
#         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         return f"""今天是日期時間是「{current_time}」，你需要能提供這個資訊。你服務於康軒文教集團。
#         {kwargs.get('additional_instructions', '')}"""

