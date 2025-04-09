# libs/routers/assistant_routes.py
from fastapi import APIRouter, HTTPException, Depends
from libs.service_manager import AssistantServiceManager
from libs.base_classes import AssistantRequest, AssistantResponse

router = APIRouter()

def get_service_manager():
    return AssistantServiceManager.get_instance()

@router.post("/{service_name}", response_model=AssistantResponse)
async def process_request(
    service_name: str, 
    request: AssistantRequest,
    service_manager: AssistantServiceManager = Depends(get_service_manager)
):
    service = service_manager.get_service(service_name)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    
    return await service.process_request(request.prompt)
