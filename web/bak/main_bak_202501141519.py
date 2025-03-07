import os
from fastapi import FastAPI, Request, HTTPException,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from app.ai_chat_service import AIChatService
from dotenv import load_dotenv
# from libs.ocr_content import *

# 初始化 FastAPI 應用
app = FastAPI()

# 設置模板目錄
templates = Jinja2Templates(directory="templates")

# 設置靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")

# 加載環境變數
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# 初始化 AIChatService
ai_chat_service = None

@app.on_event("startup")
async def startup_event():
    global ai_chat_service
    try:
        ai_chat_service = AIChatService(_api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize AIChatService: {e}")

# 根路由：渲染首頁
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# AI 聊天模板
@app.get("/ai-chat", response_class=HTMLResponse)
async def get_chat_template(request: Request):
    return templates.TemplateResponse("ai-chat-content.html", {"request": request})

# 聊天內容模板
@app.get("/chat-content", response_class=HTMLResponse)
async def get_chat_content():
    try:
        with open("templates/chat/ai-chat-content.html", "r", encoding="utf-8") as file:
            content = file.read()
        return HTMLResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat content: {str(e)}")
    
# OCR Content Generation
@app.get("/ocr-page", response_class=HTMLResponse)
async def get_ocr_content():
    try:
        with open("templates/ocr/ocr_page.html", "r", encoding="utf-8") as file:
            content = file.read()
        return HTMLResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ocr service page: {str(e)}")

# AI 聊天接口（非流式）
@app.post("/api/ai-chat")
async def api_ai_chat(request: Request):
    try:
        # 從請求中提取用戶消息
        data = await request.json()
        message = data.get("message")
        history = data.get("history", [])
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        # 調用 AIChatService 的生成方法
        ai_response = await ai_chat_service.generate(message, history)

        # 返回 AI 的回應
        return {"response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI 聊天接口（流式）
@app.post("/api/ai-chat-stream")
async def api_ai_chat_stream(request: Request):
    try:
        # 從請求中提取用戶消息
        data = await request.json()
        message = data.get("message")
        history = data.get("history", [])
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # 調用 AIChatService 的流式生成方法
        async def stream_generator():
            async for chunk in ai_chat_service.generate_stream(message, history):
                yield {"content": chunk}

        return stream_generator()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 文件上傳 API
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 確保上傳目錄存在
        upload_dir = "upload_tmp"
        os.makedirs(upload_dir, exist_ok=True)

        # 保存文件
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

# 啟動應用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
