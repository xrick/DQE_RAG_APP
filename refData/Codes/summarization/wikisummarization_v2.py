from abc import ABC, abstractmethod
from typing import Optional, Any
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import OllamaLLM
import logging

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
            raise

class ContentSummarizer(ABC):
    """內容摘要基類"""
    
    def __init__(self, llm: Any):
        self.logger = logging.getLogger(__name__)
        self.llm = llm
        
        template = """
        Role: You are an experienced and well-skilled text summarizer.
        Task:
        Please summarize the following context:
        
        {context}
        
        Please provide:
        1. Abstract: A very short overview
        2. Summarization Content (100-500 words):
           a. Most important points
           b. Extended content
        Use technical and formal style.
        """
        
        self.prompt = PromptTemplate(
            input_variables=["context"],
            template=template
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    @abstractmethod
    def get_content(self, query: str) -> Optional[str]:
        """獲取要摘要的內容"""
        pass
    
    def generate_summary(self, content: str) -> Optional[str]:
        """生成內容摘要"""
        try:
            return self.chain.run(context=content)
        except Exception as e:
            self.logger.error(f"摘要生成失敗: {str(e)}")
            return None

class WikiSummarizer(ContentSummarizer):
    """Wikipedia內容摘要器"""
    
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.wiki = WikipediaAPIWrapper()
    
    def get_content(self, query: str) -> Optional[str]:
        """從Wikipedia獲取內容"""
        try:
            return self.wiki.run(query)
        except Exception as e:
            self.logger.error(f"Wikipedia檢索失敗: {str(e)}")
            return None
    
    def summarize(self, query: str) -> Optional[str]:
        """摘要Wikipedia內容"""
        content = self.get_content(query)
        if not content:
            return None
        return self.generate_summary(content)

def main():
    # 使用示例
    llm_init = LLMInitializer()
    
    try:
        # 初始化deepseek-r1:7b模型
        llm = llm_init.init_ollama_model(
            model="deepseek-r1:7b",
            temperature=0.7
        )
        
        # 創建摘要器並使用
        wiki_summarizer = WikiSummarizer(llm)
        summary = wiki_summarizer.summarize("Artificial Intelligence")
        print("Generated Summary:")
        print(summary)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
