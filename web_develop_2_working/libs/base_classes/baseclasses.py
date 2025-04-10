from pydantic import BaseModel, Field, validator
from typing import Optional

class AssistantRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to the assistant")
    assistant_id: str = Field(..., description="The ID of the OpenAI assistant to use")
    
    @validator('assistant_id')
    def validate_assistant_id(cls, v):
        if not v.startswith('asst_'):
            raise ValueError('assistant_id must start with "asst_"')
        return v

class AssistantResponse(BaseModel):
    response: str = Field(..., description="The response from the assistant")
    thread_id: str = Field(..., description="The thread ID for future reference")
