# libs/services/base_service.py
from libs.base_classes import AssistantRequest, AssistantResponse
from fastapi import HTTPException
from openai import AsyncOpenAI
import asyncio

class BaseAssistantService:
    def __init__(self, assistant_id: str, client: AsyncOpenAI):
        self.assistant_id = assistant_id
        self.client = client

    async def call_assistant(self, prompt: str) -> tuple[str, str]:
        # 移動現有的 call_assistant 邏輯到這裡
        thread = await self.client.beta.threads.create()
        # async def call_assistant(prompt: str, assistant_id: str) -> tuple[str, str]:
        try:
            # 建立 thread
            thread = await self.client.beta.threads.create()
            
            # 新增訊息
            await self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            
            # 執行 assistant
            run = await self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # 等待完成
            while True:
                run = await self.lient.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run.status == "completed":
                    break
                elif run.status in ["failed", "expired"]:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Assistant run {run.status}"
                    )
                await asyncio.sleep(1)
            
            # 獲取回應
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            return messages.data[0].content[0].text.value, thread.id

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            return response, thread.id

    async def process_request(self, prompt: str) -> AssistantResponse:
        response, thread_id = await self.call_assistant(prompt)
        return AssistantResponse(
            response=response,
            thread_id=thread_id
        )
