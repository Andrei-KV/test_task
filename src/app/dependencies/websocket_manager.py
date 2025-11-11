from typing import List, Optional
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, text: str, websocket: WebSocket, web_link: Optional[str] = None):
        """
        Отправляет персональное сообщение.
        Если web_link предоставлен, отправляет JSON.
        В противном случае, отправляет обычный текст.
        """
        if web_link:
            payload = {
                "text": text,
                "web_link": web_link
            }
            await websocket.send_json(payload)
        else:
            await websocket.send_text(text)


manager = ConnectionManager()
