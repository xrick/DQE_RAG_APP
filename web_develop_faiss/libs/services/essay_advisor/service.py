from ..base_service import BaseAssistantService
from openai import AsyncOpenAI

# class EssayAdvisorService(BaseAssistantService):
#     def __init__(self, client: AsyncOpenAI):
#         super().__init__(
#             assistant_id="asst_uXoFZL5iarQtwvS0GZvpLhoj",  # 使用實際的 ID
#             client=client
#             )
class EssayAdvisorService(BaseAssistantService):
    def __init__(self, assistant_id: str, client: AsyncOpenAI):
        self.assistant_id=assistant_id,  # 使用實際的 ID
        self.client=client
        