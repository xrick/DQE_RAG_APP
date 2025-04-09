import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def create_assistant():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        assistant = await client.beta.assistants.create(
            name="AI Chat Assistant",
            instructions="You are a helpful AI assistant that provides clear and concise responses.",
            model="gpt-4-turbo-preview",  # or "gpt-3.5-turbo-0125"
            tools=[
                {"type": "code_interpreter"},
                {"type": "file_search"}  # Changed from 'retrieval' to 'file_search'
            ]
        )
        print(f"Assistant created successfully!")
        print(f"Assistant ID: {assistant.id}")
        return assistant.id
    except Exception as e:
        print(f"Error creating assistant: {str(e)}")
        return None

if __name__ == "__main__":
    assistant_id = asyncio.run(create_assistant())
    if assistant_id:
        print("\nAdd this to your .env file:")
        print(f"OPENAI_ASSISTANT_ID={assistant_id}")
