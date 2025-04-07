from abc import ABC, abstractmethod
from .ContentSpliter import  ContentProcessor
import logging
from typing import Optional, Any
from langchain.prompts import PromptTemplate
# from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_core.output_parsers import JsonOutputParser
import http.client
import json
import ast

class WebContentRetriever(ABC):
    """內容摘要基類"""
    def __init__(self, llm: Any):
        self.logger = logging.getLogger(__name__)
        self.llm = llm
        self.template = None
        
        # self.chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=False, return_final_only=True)
    
    @abstractmethod
    def get_content(self, query: str) -> Optional[str]:
        """獲取要摘要的內容"""
        pass
    
    @abstractmethod
    def generate_content(self, content: str) -> Optional[str]:
        """生成內容摘要"""
        pass
        

class WikiRetriever(WebContentRetriever):
    """Wikipedia內容摘要器"""
    
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.lang = 'zh' #zh-tw, en
        self.wiki = WikipediaAPIWrapper(lang=self.lang)#CustomWikiAPIWrapper()

        '''put template from WebContentRetriever to here'''
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
        '''end of template'''
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
        
    def generate_content(self, content):
        try:
            response = self.chain.invoke({"context": content},)
            # query_str = self.template.format(context=content)
            # response = self.llm.invoke(query_str)
            return response #response.get("text", "")  # 從回應中獲取文本
        except Exception as e:
            self.logger.error(f"摘要生成失敗: {str(e)}")
            return None
        
    def summarize(self, query: str) -> Optional[str]:
        """摘要Wikipedia內容"""
        content = self.get_content(query)
        if not content:
            return None
        return self.generate_content(content)
    
class GoogleSerperRetriever(WebContentRetriever):
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.conn = http.client.HTTPSConnection("google.serper.dev")
        self.headers = {
            'X-API-KEY': '3026422e86bf19796a06b5f15433e9c0cd9bd88a',
            'Content-Type': 'application/json'
        }
        self.query=None
        self.template=None
    def gen_payload(self, query:str):
        payload = json.dumps({
            "q": query
        })
        return payload

    def get_content(self, query: str) -> Optional[str]:
        """從google serper獲取內容"""
        try:
            payload = self.gen_payload(query=query)
            self.conn.request("POST", "/search", payload, self.headers)
            response = self.conn.getresponse()
            data = response.read()
            data = data.decode("utf-8")
            return data
        except Exception as e:
            self.logger.error(f"Wikipedia檢索失敗: {str(e)}")
            return None
        
    def format_googleserper_search_results(self, text_content):
        try:
            # 使用ast.literal_eval安全地評估Python字面量
            # 這對於處理Python字典和列表字符串非常有用
            data = ast.literal_eval(text_content)
            return data
        except (SyntaxError, ValueError) as e:
            print(f"解析錯誤: {e}")
            
            # 如果ast.literal_eval失敗，嘗試json解析
            try:
                # 將單引號替換為雙引號
                json_str = text_content.replace("'", '"')
                # 處理可能的特殊字符
                json_str = json_str.replace('""delete', '"delete')
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as e:
                print(f"JSON解析錯誤: {e}")
                return []

    def generate_content(self, content):
        return self.format_googleserper_search_results(content)
    
    def perform_query(self, query:str=None):
        data = self.get_content(query=query)
        data = self.generate_content(content=data)
        return data