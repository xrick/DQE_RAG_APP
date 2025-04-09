import asyncio
import os
from dotenv import load_dotenv
from app.ai_chat_service_old import AIChatService

load_dotenv()

async def test_chat_flow():
    print("Starting Chat Test Flow...")
    _api_key = os.getenv("OPENAI_API_KEY")
    # Initialize service
    service = AIChatService(
        api_key=_api_key
    )
    
    try:
        # Test 0: print the api_key
        print(f"\0. the api key is {_api_key}")
        # Test 1: Create Thread
        print("\n1. Testing Thread Creation...")
        thread_id = service.create_thread()
        print(f"Thread created: {thread_id}")
        assert thread_id is not None, "Thread creation failed"

        # Test 2: Send Message
        print("\n2. Testing Message Addition...")
        test_message = "Hello, can you help me test the system?"
        service.add_message(thread_id, test_message)
        print("Message added successfully")

        # Test 3: Generate Response
        print("\n3. Testing Response Generation...")
        print("AI Response:")
        async for chunk in service.generate(thread_id):
            print(chunk, end='', flush=True)
        print("\nResponse generation completed")

        # Test 4: Get Message History
        print("\n4. Testing Message History Retrieval...")
        messages = await service.get_messages(thread_id)
        print(f"Retrieved {len(messages)} messages")
        
        return True

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_chat_flow())
    if success:
        print("\nAll tests completed successfully!")
    else:
        print("\nTests failed!")
