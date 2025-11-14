from fastapi import WebSocket
from typing import List, Dict

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket.client.host] = websocket

    def disconnect(self, websocket: WebSocket):
        del self.active_connections[websocket.client.host]

    async def send_personal_message(
        self,
        text: str,
        websocket: WebSocket,
        web_link: str = None,
        title: str = None,
        page_numbers: list[int] = None,
        sections: list[str] = None
    ):
        if web_link:
            # Отправляем JSON, если есть веб-ссылка
            response = {
                "text": text,
                "web_link": web_link,
                "title": title,
                "page_numbers": page_numbers or [],
                "sections": sections or []
            }
            await websocket.send_json(response)
        else:
            # Отправляем обычный текст для простых сообщений (например, "Идёт обработка...")
            await websocket.send_text(text)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()
