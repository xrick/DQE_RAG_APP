from fastapi import APIRouter, Request, Depends, WebSocket
from typing import Optional
from ..ai_chat_service_old import AIChatService

router = APIRouter(prefix="/aichat")

async def get_ai_chat_service():
    """依賴注入：獲取AI聊天服務實例"""
    return AIChatService()

@router.get("/models")
async def get_models(service: AIChatService = Depends(get_ai_chat_service)):
    """獲取可用的模型列表"""
    return service.model_manager.get_available_models()

@router.get("/newThread")
async def new_thread(
    request: Request,
    service: AIChatService = Depends(get_ai_chat_service)
):
    """創建新的對話線程"""
    thread_id = await service.create_thread()
    request.session["AIThreadId"] = thread_id
    request.session["storage-service"] = ""  # 清除 storage-service cookie
    return {"Thread": {"id": thread_id}}

@router.get("/putMessage")
async def put_message(
    request: Request,
    content: str,
    service: AIChatService = Depends(get_ai_chat_service)
):
    """添加用戶消息"""
    thread_id = request.session.get("AIThreadId")
    if not thread_id:
        thread_id = await service.create_thread()
        request.session["AIThreadId"] = thread_id
    
    return await service.add_message(thread_id, content)

@router.get("/messages")
async def get_messages(
    request: Request,
    service: AIChatService = Depends(get_ai_chat_service)
):
    """獲取歷史消息"""
    thread_id = request.session.get("AIThreadId")
    if not thread_id:
        return {"error": "No active thread"}
    
    messages = await service.get_messages(thread_id)
    return {"Thread": thread_id, "msg": messages}

@router.get("/run")
async def run_chat(
    request: Request,
    model_id: str = "assistant-default",
    service: AIChatService = Depends(get_ai_chat_service)
):
    """運行AI對話"""
    thread_id = request.session.get("AIThreadId")
    if not thread_id:
        thread_id = await service.create_thread()
        request.session["AIThreadId"] = thread_id
    
    return await service.generate(
        thread_id,
        model_id,
        account=request.session.get("account"),
        displayName=request.session.get("displayName"),
        teacherId=request.session.get("teacherId")
    )

@router.get("/runStatus")
async def run_status(
    request: Request,
    service: AIChatService = Depends(get_ai_chat_service)
):
    """獲取運行狀態"""
    thread_id = request.session.get("AIThreadId")
    run_id = request.session.get("RunId")
    if not thread_id or not run_id:
        return {"error": "No active thread or run"}
    
    return await service.get_run_status(thread_id, run_id)

@router.get("/runCancel")
async def run_cancel(
    request: Request,
    service: AIChatService = Depends(get_ai_chat_service)
):
    """取消運行"""
    thread_id = request.session.get("AIThreadId")
    run_id = request.session.get("RunId")
    if not thread_id or not run_id:
        return {"error": "No active thread or run"}
    
    return await service.cancel_run(thread_id, run_id)
