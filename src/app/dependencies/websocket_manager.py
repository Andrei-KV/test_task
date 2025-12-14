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

    async def send_personal_message(self, text: str, websocket: WebSocket, 
                                  web_link: Optional[str] = None,  
                                  title: Optional[str] = None, 
                                  page_numbers: list[int] = None, 
                                  documents: list[dict] = None,
                                  is_loading: bool = False):
        """
        Отправляет персональное сообщение.
        Поддерживает отправку списка документов.
        Для совместимости: если передан documents, но нет web_link, 
        использует первый документ для создания основной кнопки.
        """
        if is_loading:
            # Отправляем JSON с флагом loading
            payload = {
                "text": text,
                "is_loading": True
            }
            await websocket.send_json(payload)
            return

        # Формируем payload для обычного сообщения
        payload = {"text": text}

        # Логика обратной совместимости для кнопок (ОТКЛЮЧЕНА, так как создает дубликаты на фронте)
        # if not web_link and documents and len(documents) > 0:
        #     web_link = documents[0].get('web_link')
        #     title = documents[0].get('title')
            # Можно также собрать страницы из первого документа, если нужно
            # page_numbers = list(documents[0].get('pages', []))

        if web_link:
            payload["web_link"] = web_link
            payload["title"] = title
        
        if page_numbers:
            payload["page_numbers"] = page_numbers
            
        if documents:
            payload["documents"] = documents

        # Если есть структурированные данные (ссылка или документы), отправляем JSON
        if web_link or documents:
            await websocket.send_json(payload)
        else:
            # Иначе отправляем просто текст (как раньше)
            await websocket.send_text(text)


manager = ConnectionManager()
