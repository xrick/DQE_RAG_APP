import os
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from app.ai_chat_service_deprecated_202501091217 import AIChatService
from pathlib import Path
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

app = FastAPI()

# 設置模板目錄
templates = Jinja2Templates(directory="templates")
# 設置靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize AI-Chatservice
# load .env file to environment
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
print(f"api_key is {_api_key}")
ai_chat_service = None;

@app.on_event("startup")
async def startup_event():
    global ai_chat_service
    ai_chat_service = await AIChatService(_api_key).initialize()


# @app.get("/")
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/chat")
# async def chat(request: Request):
#     return templates.TemplateResponse("ai-chat.html", {"request": request})
@app.get("/ai-chat", response_class=HTMLResponse)
async def get_chat_template(request: Request):
    return templates.TemplateResponse("ai-chat.html", {"request": request})

@app.post("/api/chat/thread")
async def create_thread():
    try:
        thread_id = await ai_chat_service.create_thread()
        return {"thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/message")
async def chat_message(
    thread_id: str,
    message: str,
    request: Request
):
    async def event_generator():
        try:
            async for response in ai_chat_service.generate(thread_id, message):
                yield f"data: {json.dumps({'content': response})}\n\n"
            yield "event: thread.run.completed\ndata: {}\n\n"
        except Exception as e:
            # logger.error(f"Error in chat_message: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.get("/api/chat/history/{thread_id}")
async def get_chat_history(thread_id: str):
    try:
        messages = await ai_chat_service.get_messages(thread_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
