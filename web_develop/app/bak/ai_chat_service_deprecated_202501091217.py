from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from openai import OpenAI  # Using regular OpenAI
from .base_classes import BaseLLMService, BasePromptTemplate, ModelManager

class AIChatService(BaseLLMService):
    """AI聊天服務實現"""
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.client = OpenAI(api_key=self.api_key)  # Regular OpenAI client
        self.model = "gpt-4"  # or "gpt-3.5-turbo"
        self.thread_messages = {}
        self.system_message = {
            "role": "system",
            "content": "You are a helpful assistant for 康軒文教集團. You should provide accurate and educational information."
        }

    async def initialize(self):
        """初始化服務"""
        return self

    async def create_thread(self) -> str:
        """創建新的對話線程"""
        import uuid
        thread_id = str(uuid.uuid4())
        self.thread_messages[thread_id] = [self.system_message]
        return thread_id

    async def add_message(self, thread_id: str, content: str, role: str = "user") -> Dict:
        """添加用戶消息到對話歷史"""
        if thread_id not in self.thread_messages:
            self.thread_messages[thread_id] = [self.system_message]
        
        message = {"role": role, "content": content}
        self.thread_messages[thread_id].append(message)
        return {"thread_id": thread_id, "message": message}

    async def get_messages(self, thread_id: str, limit: int = 30) -> List[Dict]:
        """獲取歷史消息"""
        if thread_id not in self.thread_messages:
            return []
        return self.thread_messages[thread_id][-limit:]

    async def generate(self, thread_id: str, message: str) -> StreamingResponse:
        """生成AI回應（流式）"""
        # Add user message to history
        await self.add_message(thread_id, message)
        
        # Get conversation history for this thread
        messages = self.thread_messages.get(thread_id, [self.system_message])
        
        # Create completion with streaming
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        async def generate_stream():
            collected_message = []
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_message.append(content)
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            # Add assistant's complete response to history
            full_response = "".join(collected_message)
            await self.add_message(thread_id, full_response, role="assistant")
            yield "event: thread.run.completed\ndata: {}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    async def get_run_status(self, thread_id: str) -> Dict:
        """獲取對話狀態"""
        if thread_id not in self.thread_messages:
            return {"status": "not_found"}
        return {
            "status": "active",
            "message_count": len(self.thread_messages[thread_id])
        }

    async def cancel_run(self, thread_id: str) -> Dict:
        """取消對話"""
        if thread_id in self.thread_messages:
            del self.thread_messages[thread_id]
        return {"status": "cancelled", "thread_id": thread_id}
