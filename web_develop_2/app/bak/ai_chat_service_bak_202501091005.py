import os
import logging
import asyncio
from openai import AsyncOpenAI
from openai import OpenAI

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AIChatService:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
         # Create an assistant if it doesn't exist
        self.assistant_id = None;#self._create_assistant()
        logger.debug("AIChatService initialized")

    async def initialize(self):
        """Initialize the service and create the assistant"""
        if not self.assistant_id:
            self.assistant_id = await self._create_assistant()
        print(f"assistant_id is {self.assistant_id}");
        return self

    async def _create_assistant(self):
        """Create an OpenAI assistant"""
        try:
            assistant = await self.client.beta.assistants.create(
                name="Chat Assistant",
                instructions="You are a helpful AI assistant.",
                model="gpt-4-turbo-preview",  # or "gpt-3.5-turbo-1106"
                tools=[]  # Add any tools if needed
            )
            logger.info(f"Assistant created with ID: {assistant.id}")
            return assistant
        except Exception as e:
            logger.error(f"Error creating assistant: {e}")
            raise
        
    async def create_thread(self):
        """Create a new conversation thread"""
        try:
            thread = await self.client.beta.threads.create()
            return thread.id
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            raise
            
    async def add_message(self, thread_id: str, content: str):
        try:
            logger.debug(f"Adding message to thread {thread_id}")
            message = await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=content
            )
            logger.debug(f"Message added: {message.id}")
            return message.id
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise
            
    async def generate(self, thread_id: str, user_message: str):
        """Generate response using the assistant"""
        try:
            # Add the user message to the thread
            await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_message
            )

            # Create a run with the assistant
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id.id
            )

            # Wait for the run to complete and get messages
            while True:
                run_status = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                if run_status.status == 'completed':
                    # Get the messages
                    messages = await self.client.beta.threads.messages.list(
                        thread_id=thread_id
                    )
                    
                    # Get the latest assistant message
                    for message in messages:
                        print(f"message1:{message[0]}\n\n")
                        print(f"message2:{message[1]}")
                        if message[1].content.role == "assistant":
                            yield message.content[0].text.value
                            break
                    break
                
                elif run_status.status == 'failed':
                    logger.error(f"Run failed: {run_status.last_error}")
                    yield "Sorry, there was an error generating the response."
                    break
                
                await asyncio.sleep(1)  # Wait before checking again

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            yield f"Error: {str(e)}"


    async def get_messages(self, thread_id: str):
        try:
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id
            )
            return [
                {
                    'role': msg[1].content.role,
                    'content': msg[1].content[0].text.value,
                    'created_at': msg.created_at
                }
                for msg in messages.data
            ]
        except Exception as e:
            logger.error(f"Error getting messages: {str(e)}")
            raise
