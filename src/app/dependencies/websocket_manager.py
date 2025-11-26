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

    async def send_personal_message(self, text: str, websocket: WebSocket, web_link: Optional[str] = None,  title: Optional[str] = None, page_numbers: list[int] = None, is_loading: bool = False):
        """
        Отправляет персональное сообщение.
        Если web_link предоставлен, отправляет JSON.
        Если is_loading=True, отправляет специальное сообщение-индикатор загрузки.
        В противном случае, отправляет обычный текст.
        """
        if is_loading:
            # Отправляем JSON с флагом loading
            payload = {
                "text": text,
                "is_loading": True
            }
            await websocket.send_json(payload)
        elif web_link:
            payload = {
                "text": text,
                "web_link": web_link,
                'title': title
            }
            if page_numbers:
                payload["page_numbers"] = page_numbers
            await websocket.send_json(payload)
        else:
            await websocket.send_text(text)


manager = ConnectionManager()
