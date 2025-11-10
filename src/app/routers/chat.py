from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from ..dependencies.websocket_manager import manager
from ..dependencies.rag import get_rag_service
from src.services.rag_service import RAGService

router = APIRouter(tags=["chat"])

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    rag_service: RAGService = Depends(get_rag_service),
):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            answer, web_link = await rag_service.aquery(data)
            await manager.send_personal_message(answer, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
