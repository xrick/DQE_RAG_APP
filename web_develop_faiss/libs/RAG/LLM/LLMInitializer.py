import logging
from langchain_ollama import OllamaLLM

class LLMInitializer:
    """LLM模型初始化類"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def init_ollama_model(
        self, 
        model: str = "deepseek-r1:7b",
        base_url: str = "http://localhost:11434",
        **kwargs
    ) -> OllamaLLM:
        """初始化OllamaLLM模型
        
        Args:
            model: Ollama模型名稱
            base_url: Ollama服務URL
            **kwargs: 額外的模型參數
            
        Returns:
            初始化後的OllamaLLM實例
        """
        try:
            return OllamaLLM(
                model=model,
                base_url=base_url,
                streaming=True,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"OllamaLLM模型初始化失敗: {str(e)}")
            raise