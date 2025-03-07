import os
from fastapi import FastAPI, Request, HTTPException,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.ai_chat_service import AIChatService
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
from libs.routers.assistant_routes import router as assistant_router
from libs.service_manager import AssistantServiceManager    
from libs.base_classes import AssistantRequest, AssistantResponse
import logging
# from libs.ocr_content import *
# OCR Libraries
# import pytesseract
# from PIL import Image
# import io
# import pdf2image

# 初始化 FastAPI 應用
app = FastAPI()

# CORS 設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設置具體的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設置模板目錄
templates = Jinja2Templates(directory="templates")

# 設置靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")
# 在 app 初始化後添加
app.include_router(assistant_router, prefix="/assistant")

# 加載環境變數
load_dotenv()

_api_key = "ihfashi42423#@@$@";#os.getenv("OPENAI_API_KEY")
logging.info("openai api key: " + _api_key);
# if not _api_key:
#     raise ValueError("OPENAI_API_KEY environment variable is not set")

# 添加配置
ASSISTANT_CONFIG = {
    "ESSAY_ADVISOR_ID": "asdfasgagafsadfad",#os.getenv("ESSAY_ADVISOR_ASSISTANT_ID"),
    "K9_HELPER_ID": "sadfdsfasdfsdadfad"#os.getenv("K9_HELPER_ASSISTANT_ID")
}
#    配置
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # 安裝tes
# 初始化 AIChatService
# ai_chat_service = None
# initialize AsyncOpenAI
# client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# client = None;
# service_manager = None;
"""
pure functions
"""
# def ocr_image(image):
#     """
#     對圖片進行識別
#     :param image: PIL Image
#     :return: 識別出的文字
#     """
#     try:
#         # 使用中文和英文
#         text = pytesseract.image_to_string(image, lang='chi_tra')
#         return text.strip()
#     except Exception as e:
#         print(f"OCR識別錯誤: {e}")
#         return ""

# def ocr_pdf(pdf_file):
#     """
#     对PDF文件執行OCR
#     :param pdf_file: 要識別的PDF文件
#     :return: 識别出的文字
#     """
#     try:
#         # 將pdf轉為圖片
#         images = pdf2image.convert_from_bytes(pdf_file.file.read())
        
#         # 對每頁執行ocr
#         all_text = []
#         for image in images:
#             page_text = ocr_image(image)
#             all_text.append(page_text)
        
#         return "\n\n".join(all_text)
#     except Exception as e:
#         print(f"PDF OCR識別錯誤: {e}")
#         return ""

"""
end of pure function
"""

# @app.post("/perform_ocr")
# async def perform_ocr(file: UploadFile = File(...)):
#     """
#     OCR Web-API
#     """
#     try:
#         # check the if the file is null
#         if not file.filename:
#             raise HTTPException(status_code=400, detail="未選擇文件")

#         # get document type
#         content_type = file.content_type

#         # 根據文件類別執行對應的OCR方法
#         if content_type.startswith('image/'):
#             # 對圖片進行處理
#             image = Image.open(io.BytesIO(await file.read()))
#             extracted_text = ocr_image(image)
#         elif content_type == 'application/pdf':
#             # 對pdf文件進行處理
#             extracted_text = ocr_pdf(file)
#         else:
#             raise HTTPException(status_code=400, detail="不支援的文件類型")

#         # return results
#         return {
#             "text": extracted_text,
#             "success": bool(extracted_text),
#             "filename": file.filename
#         }

#     except Exception as e:
#         print(f"OCR處理錯號: {e}")
#         raise HTTPException(status_code=500, detail=f"OCR處理失敗: {str(e)}")

@app.on_event("startup")
async def startup_event():
    global ai_chat_service
    global service_manager
    global async_client
    try:
        logging.info("start to initialize services.......");
        ai_chat_service = AIChatService()#AIChatService(_api_key)
        # 初始化助手服務管理器
        async_client = AsyncOpenAI(api_key=_api_key);
        service_manager = AssistantServiceManager.initialize(async_client)
        service_manager.initialize_services(ASSISTANT_CONFIG)
        logging.info("chat, essay-advisor, k9-helper services are initialized!");
    except Exception as e:
        logging.error(f"Failed to run in startup_event: {e}");
        raise RuntimeError(f"Failed to initialize AIChatService: {e}")

# 根路由：渲染首頁
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# AI 聊天模板
# @app.get("/ai-chat", response_class=HTMLResponse)
# async def get_chat_template(request: Request):
#     return templates.TemplateResponse("ai-chat-content.html", {"request": request})

# Essay Advisor 模板
@app.get("/essay-advisor", response_class=HTMLResponse)
async def get_essay_advisor_template():
    # return templates.TemplateResponse("templates/essay_advisor/content.html", {"request": request})
    try:
        with open("templates/essay_advisor/content.html", "r", encoding="utf-8") as file:
            content = file.read()
        return HTMLResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat content: {str(e)}")

# K9 Helper 模板
@app.get("/k9-helper", response_class=HTMLResponse)
async def get_k9_helper_template():
    # return templates.TemplateResponse("templates/k9_helper/content.html", {"request": request})
    try:
        with open("templates/k9_helper/content.html", "r", encoding="utf-8") as file:
            content = file.read()
        return HTMLResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat content: {str(e)}")


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
        return {"response": ai_response}; 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @app.post("/api/ai-chat")
# async def api_ai_chat(request: Request):
#     try:
#         # 從請求中提取用戶消息
#         data = await request.json()
#         message = data.get("message")
#         history = data.get("history", [])
#         if not message:
#             raise HTTPException(status_code=400, detail="Message is required")
#         # 調用 AIChatService 的生成方法
#         ai_response = await ai_chat_service.generate(message, history)

#         # 返回 AI 的回應
#         return {"response": ai_response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

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
    
# Essay Advisor API
@app.post("/api/essay-advisor")
async def essay_advisor_endpoint(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "")
        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )
        response = await service_manager.handle_essay_advisor(message)
        return {"response": response}
    except Exception as e:
        logging.error(f"Essay Advisor error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

# K9 Helper API
@app.post("/api/k9-helper")
async def k9_helper_endpoint(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "")
        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )
        response = await service_manager.handle_k9_helper(message)
        return {"response": response}
    except Exception as e:
        logging.error(f"K9 Helper error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

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

"""
Assistant API Call
"""
async def call_assistant(prompt: str, assistant_id: str) -> tuple[str, str]:
    try:
        # 建立 thread
        thread = await client.beta.threads.create()
        
        # 新增訊息
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        
        # 執行 assistant
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        
        # 等待完成
        while True:
            run = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run.status == "completed":
                break
            elif run.status in ["failed", "expired"]:
                raise HTTPException(
                    status_code=500,
                    detail=f"Assistant run {run.status}"
                )
            await asyncio.sleep(1)
        
        # 獲取回應
        messages = await client.beta.threads.messages.list(
            thread_id=thread.id
        )
        
        return messages.data[0].content[0].text.value, thread.id

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant", response_model=AssistantResponse)
async def ask_assistant(request: AssistantRequest):
    """
    端點用於與 OpenAI Assistant 互動
    """
    try:
        response, thread_id = await call_assistant(
            request.prompt,
            request.assistant_id
        )
        
        return AssistantResponse(
            response=response,
            thread_id=thread_id
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 錯誤處理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# 啟動應用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
