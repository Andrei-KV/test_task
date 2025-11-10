from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..dependencies.websocket_manager import manager

router = APIRouter(tags=["chat"])

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Logic for RAG and context will go here
            await manager.send_personal_message(f"Ответ: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # Optionally, notify other users
        # await manager.broadcast(f"Пользователь #{client_id} покинул чат")
