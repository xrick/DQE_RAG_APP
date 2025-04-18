import asyncio
import os
from dotenv import load_dotenv
from app.ai_chat_service_deprecated_202501091217 import AIChatService

load_dotenv()

async def test_chat():
    service = AIChatService(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        # Create thread
        print("Creating thread...")
        thread_id = await service.create_thread()
        print(f"Thread created: {thread_id}")
        
        # Add message
        print("\nSending message...")
        await service.add_message(thread_id, "Hello, can you help me test if this is working?")
        
        # Generate response
        print("\nWaiting for response...")
        async for chunk in service.generate(thread_id):
            print(chunk, end='', flush=True)
        print("\n\nResponse complete!")
        
        # Get message history
        messages = await service.get_messages(thread_id)
        print("\nMessage history:")
        for msg in messages:
            print(f"{msg['role']}: {msg['content'][:100]}...")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    print("Starting chat test...")
    print(f"API Key (first 4 chars): {os.getenv('OPENAI_API_KEY')[:4]}...")
    # print(f"Assistant ID: {os.getenv('OPENAI_ASSISTANT_ID')}")
    
    asyncio.run(test_chat())
