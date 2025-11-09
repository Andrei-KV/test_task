from typing import Dict, List, Union
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """
    Класс для управления активными WebSocket-соединениями.
    Хранит словарь (Dict), где ключ — это уникальный ID клиента, 
    а значение — сам объект WebSocket.
    """
    def __init__(self):
        # Хранит все активные соединения: {client_id: WebSocket}
        self.active_connections: Dict = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Принимает новое соединение и добавляет его в словарь."""
        # 1. Принять соединение от клиента
        await websocket.accept()
        
        # 2. Сохранить соединение
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        """Удаляет соединение при отключении клиента."""
        # Используем.pop() для удаления по ключу
        if client_id in self.active_connections:
            self.active_connections.pop(client_id, None)

    async def send_personal_message(self, message: Union[str, dict], client_id: str):
        """Отправляет сообщение конкретному клиенту по его ID."""
        websocket = self.active_connections.get(client_id)
        if websocket:
            # FastAPI/Starlette может автоматически сериализовать словарь в JSON
            if isinstance(message, dict):
                await websocket.send_json(message)
            else:
                await websocket.send_text(message)

    async def broadcast(self, message: Union[str, dict]):
        """Отправляет сообщение всем активным клиентам."""
        for connection in self.active_connections.values():
            if isinstance(message, dict):
                await connection.send_json(message)
            else:
                await connection.send_text(message)

# Создаем единственный экземпляр менеджера, который будет использоваться
# во всем приложении (паттерн 'Singleton').
manager = ConnectionManager()