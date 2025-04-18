# libs/service_manager.py
from typing import Dict
from openai import AsyncOpenAI
from enum import Enum
from typing import Optional
import logging
from libs.services.essay_advisor.service import EssayAdvisorService
from libs.services.k9_helper.service import K9HelperService
from libs.services.base_service import BaseAssistantService

class ServiceType(Enum):
    AI_CHAT = "ai_chat"
    ESSAY_ADVISOR = "essay_advisor"
    K9_HELPER = "k9_helper"

class AssistantServiceManager:
    _instance = None
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.services: Dict[str, BaseAssistantService] = {}
        self.llm = None  # LLM 實例
        self._initialize_prompts()
        
    @classmethod
    def get_instance(cls) -> 'AssistantServiceManager':
        if cls._instance is None:
            raise RuntimeError("ServiceManager not initialized")
        return cls._instance
    
    @classmethod
    def initialize(cls, client: AsyncOpenAI) -> 'AssistantServiceManager':
        if cls._instance is None:
            cls._instance = cls(client)
        return cls._instance
    
    def _initialize_prompts(self):
        self.prompts = {
            ServiceType.AI_CHAT: """You are a helpful AI assistant...""",  # 保持原有的
            
            ServiceType.ESSAY_ADVISOR: """您是一位專業的寫作顧問。
            您的角色是協助使用者改進他們的寫作：
            1. 分析文章結構和文字流暢度
            2. 提供用字遣詞的改進建議
            3. 針對文章清晰度和表達效果提供具體建議
            4. 協助掌握學術寫作規範
            
            請提供詳細且具建設性的回饋，同時保持支持和鼓勵的語氣。
            
            在回覆時，請：
            - 先肯定文章的優點
            - 分點說明可以改進的地方
            - 提供具體的修改建議
            - 使用易於理解的說明方式
            
            請使用繁體中文回覆。
            
            使用者的訊息：{message}""",
            
            ServiceType.K9_HELPER: """您是一位專業的 K9 商城購物顧問。
            您的角色是：
            1. 協助使用者找到合適的商品
            2. 解答關於商品特色和規格的問題
            3. 根據使用者需求提供商品建議
            4. 說明不同商品之間的差異與比較
            
            在回答時，請：
            - 提供準確的商品資訊
            - 以客觀的角度分析比較
            - 考慮使用者的實際需求
            - 適時提供相關商品推薦
            
            請使用親切專業的態度，以繁體中文回覆。
            
            使用者的訊息：{message}"""
        }

    async def _generate_response(self, prompt: str) -> str:
        """使用 LLM 生成回應"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",  # 或其他模型
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM 回應生成失敗: {str(e)}")
            raise e

    async def handle_essay_advisor(self, message: str) -> str:
        try:
            prompt = self.prompts[ServiceType.ESSAY_ADVISOR].format(message=message)
            response = await self._generate_response(prompt)
            return response
        except Exception as e:
            logging.error(f"Error in essay advisor: {str(e)}")
            return "抱歉，處理您的寫作建議時出現問題。請稍後再試。"

    async def handle_k9_helper(self, message: str) -> str:
        try:
            prompt = self.prompts[ServiceType.K9_HELPER].format(message=message)
            response = await self._generate_response(prompt)
            return response
        except Exception as e:
            logging.error(f"Error in k9 helper: {str(e)}")
            return "抱歉，處理您的商城諮詢時出現問題。請稍後再試。"

    def initialize_services(self, config: dict):
        """
        初始化所有助手服務
        """
        self.services = {
            "essay_advisor": EssayAdvisorService(
                client=self.client,
                assistant_id=config.get("ESSAY_ADVISOR_ID")
            ),
            "k9_helper": K9HelperService(
                client=self.client,
                assistant_id=config.get("K9_HELPER_ID")
            )
        }
    
    def get_service(self, service_name: str) -> BaseAssistantService:
        return self.services.get(service_name)
