# libs/services/k9_helper/service.py
from ..base_service import BaseAssistantService
from openai import AsyncOpenAI

# class K9HelperService(BaseAssistantService):
#     def __init__(self, client: AsyncOpenAI):
#         super().__init__(
#             assistant_id="asst_ix5YLGFEiQzm5QMY52CjvKKQ",  # 使用實際的 ID
#             client=client
#         )

class K9HelperService(BaseAssistantService):
    def __init__(self, assistant_id: str, client: AsyncOpenAI):
        self.assistant_id=assistant_id,  # 使用實際的 ID
        self.client=client