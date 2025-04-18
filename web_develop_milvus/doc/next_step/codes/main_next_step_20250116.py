# main.py
from fastapi import File, UploadFile
import aiofiles
import os
from typing import List
import base64

# 配置臨時文件目錄
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/api/ai-chat")
async def ai_chat_endpoint(
    message: str = None,
    images: List[UploadFile] = File(None),
    audio: UploadFile = File(None)
):
    try:
        # 處理文字訊息
        if message is None and not images and not audio:
            return JSONResponse(
                status_code=400,
                content={"error": "No input provided"}
            )

        # 處理圖片
        image_data = []
        if images:
            for img in images:
                if img.content_type not in ["image/jpeg", "image/png"]:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Unsupported image format"}
                    )
                content = await img.read()
                base64_image = base64.b64encode(content).decode('utf-8')
                image_data.append(base64_image)

        # 處理音頻
        audio_text = None
        if audio:
            if audio.content_type not in ["audio/mpeg", "audio/wav", "audio/mp4"]:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Unsupported audio format"}
                )
            
            # 保存臨時文件
            temp_path = os.path.join(TEMP_DIR, audio.filename)
            async with aiofiles.open(temp_path, 'wb') as out_file:
                content = await audio.read()
                await out_file.write(content)

            # 使用 Whisper API 轉換音頻為文字
            audio_text = await service_manager.transcribe_audio(temp_path)
            
            # 清理臨時文件
            os.remove(temp_path)

        # 組合所有輸入
        response = await service_manager.handle_ai_chat(
            message=message,
            images=image_data if image_data else None,
            audio_text=audio_text
        )
        
        return {"response": response}

    except Exception as e:
        logger.error(f"AI Chat error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

##################################################################################
# main.py
from typing import List, Optional
from pydantic import BaseModel

class MessageContent(BaseModel):
    type: str  # 'text', 'image', 'audio', 'code'
    content: str
    metadata: Optional[dict] = None  # 額外資訊，如 mime type, 大小等

class ChatResponse(BaseModel):
    message_id: str
    timestamp: str
    contents: List[MessageContent]
    
@app.post("/api/ai-chat")
async def ai_chat_endpoint(
    message: str = None,
    images: List[UploadFile] = File(None),
    audio: UploadFile = File(None)
):
    try:
        # ... 前面的處理邏輯保持不變 ...

        # 生成結構化回應
        response = await service_manager.handle_ai_chat(
            message=message,
            images=image_data if image_data else None,
            audio_text=audio_text
        )
        
        # 格式化回應
        formatted_response = ChatResponse(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            contents=response  # 現在 response 會是結構化的數據
        )
        
        return formatted_response.dict()

    except Exception as e:
        logger.error(f"AI Chat error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )