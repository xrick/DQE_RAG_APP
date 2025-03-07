import os
from fastapi import FastAPI, Request, HTTPException,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.ai_chat_service import AIChatService
from dotenv import load_dotenv
from libs.base_classes import AssistantRequest, AssistantResponse
# from libs.ocr_content import *

# OCR Libraries
import pytesseract
from PIL import Image
import io
import pdf2image

# 初始化 FastAPI 應用
app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設置模板目錄
templates = Jinja2Templates(directory="templates")

# 設置靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")

# 加載環境變數
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
#    配置
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # 安裝tes
# 初始化 AIChatService
ai_chat_service = None

"""
pure functions
"""
def ocr_image(image):
    """
    對圖片進行識別
    :param image: PIL Image
    :return: 識別出的文字
    """
    try:
        # 使用中文和英文
        text = pytesseract.image_to_string(image, lang='chi_tra')
        return text.strip()
    except Exception as e:
        print(f"OCR識別錯誤: {e}")
        return ""

def ocr_pdf(pdf_file):
    """
    对PDF文件執行OCR
    :param pdf_file: 要識別的PDF文件
    :return: 識别出的文字
    """
    try:
        # 將pdf轉為圖片
        images = pdf2image.convert_from_bytes(pdf_file.file.read())
        
        # 對每頁執行ocr
        all_text = []
        for image in images:
            page_text = ocr_image(image)
            all_text.append(page_text)
        
        return "\n\n".join(all_text)
    except Exception as e:
        print(f"PDF OCR識別錯誤: {e}")
        return ""

"""
end of pure function
"""

@app.post("/perform_ocr")
async def perform_ocr(file: UploadFile = File(...)):
    """
    OCR Web-API
    """
    try:
        # check the if the file is null
        if not file.filename:
            raise HTTPException(status_code=400, detail="未選擇文件")

        # get document type
        content_type = file.content_type

        # 根據文件類別執行對應的OCR方法
        if content_type.startswith('image/'):
            # 對圖片進行處理
            image = Image.open(io.BytesIO(await file.read()))
            extracted_text = ocr_image(image)
        elif content_type == 'application/pdf':
            # 對pdf文件進行處理
            extracted_text = ocr_pdf(file)
        else:
            raise HTTPException(status_code=400, detail="不支援的文件類型")

        # return results
        return {
            "text": extracted_text,
            "success": bool(extracted_text),
            "filename": file.filename
        }

    except Exception as e:
        print(f"OCR處理錯號: {e}")
        raise HTTPException(status_code=500, detail=f"OCR處理失敗: {str(e)}")

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
