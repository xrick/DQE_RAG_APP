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
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"OllamaLLM模型初始化失敗: {str(e)}")

# def _query_llm(self, query: str) -> str:
#     """連接實際的LLM API"""
    
    
#     # 初始化LLM客戶端

    
#     # 準備系統指令和用戶消息
#     messages = [
#         SystemMessage(content=f"使用{st.session_state.search_mode}模式搜索，閾值設為{st.session_state.threshold}"),
#         HumanMessage(content=query)
#     ]
    
#     # 獲取回應
#     response = chat_model(messages)
#     return response.content