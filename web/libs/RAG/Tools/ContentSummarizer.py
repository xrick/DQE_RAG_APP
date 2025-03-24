from abc import ABC, abstractmethod
from .ContentSpliter import  ContentProcessor
import logging
from typing import Optional, Any
from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.output_parsers import JsonOutputParser

class ContentSummarizer(ABC):
    """內容摘要基類"""
    def __init__(self, llm: Any):
        self.logger = logging.getLogger(__name__)
        self.llm = llm
        
        self.template = """
        Role: You are an experienced and well-skilled text summarizer.
        Task:
        Please summarize the following context:
        
        {context}
        
        Please provide:
        1. Abstract: A very short overview
        2. Summarization Content (100-500 words):
           a. Most important points
           b. Extended content
        3. Use technical and formal style.
        """
        self.prompt = PromptTemplate(
            input_variables=["context"],
            template=self.template
        )
        self.content_parser = ContentProcessor()
        self.chain = self.prompt | self.llm
        # self.chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=False, return_final_only=True)

    # def llm_invoke
    
    @abstractmethod
    def get_content(self, query: str) -> Optional[str]:
        """獲取要摘要的內容"""
        pass
    
    def generate_summary(self, content: str) -> Optional[str]:
        """生成內容摘要"""
        try:
            response = self.chain.invoke({"context": content},)
            # query_str = self.template.format(context=content)
            # response = self.llm.invoke(query_str)
            return response #response.get("text", "")  # 從回應中獲取文本
        except Exception as e:
            self.logger.error(f"摘要生成失敗: {str(e)}")
            return None
        

class WikiSummarizer(ContentSummarizer):
    """Wikipedia內容摘要器"""
    
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.lang = 'zh' #zh-tw, en
        self.wiki = WikipediaAPIWrapper(lang=self.lang)#CustomWikiAPIWrapper()
        
        # self.wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(), )
        # self.wiki = WikipediaQueryRun(CustomWikiAPIWrapper())
    
    def get_content(self, query: str) -> Optional[str]:
        """從Wikipedia獲取內容"""
        try:
            # filtered_content = self.content_parser.split_pages(self.wiki.run(query))
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