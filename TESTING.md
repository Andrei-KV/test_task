# Testing Plan

This document outlines the testing strategy for the RAG chatbot application. All tests should be implemented using the `pytest` framework, with `httpx.AsyncClient` for API-level testing.

## 1. Unit Testing (Service Layer)

**Goal:** Verify the logic of individual, isolated modules without involving FastAPI, databases, or external services like LLMs.

**Tools:** `pytest`, `unittest.mock`

**Example:** Test the `ContextManagerService` to ensure it correctly retrieves the last N messages and manages the clarification counter.

```python
# tests/unit/test_context_manager.py
import pytest
from unittest.mock import AsyncMock
from src.services.context_manager import ContextManagerService

@pytest.mark.asyncio
async def test_get_context_window():
    # Arrange
    mock_redis_service = AsyncMock()
    mock_redis_service.get_history.return_value = [
        {"role": "user", "content": "Message 1"},
        {"role": "bot", "content": "Message 2"},
        {"role": "user", "content": "Message 3"},
        {"role": "bot", "content": "Message 4"},
    ]
    context_manager = ContextManagerService(redis_service=mock_redis_service)

    # Act
    context_window = await context_manager.get_context_window(session_id="test_session", window_size=2)

    # Assert
    assert len(context_window) == 2
    assert context_window[0]["content"] == "Message 3"
    assert context_window[1]["content"] == "Message 4"
```

## 2. Testing Dependencies (Mocking DI)

**Goal:** Verify that FastAPI endpoints process requests correctly by substituting real, slow, or external dependencies (LLM, Qdrant, Redis) with mock objects.

**Tools:** `pytest`, `httpx.AsyncClient`, `fastapi.testclient`, FastAPI's `app.dependency_overrides`

**Example:** Test the WebSocket endpoint by mocking the `get_rag_service` dependency to return a fixed response.

```python
# tests/dependencies/test_chat_endpoint.py
import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from src.app.dependencies.rag import get_rag_service

async def mock_rag_service():
    class MockRAGService:
        async def aquery(self, query: str, low_precision: bool = False):
            return "Mocked RAG response", "http://mock.link", 0.9
    yield MockRAGService()

app.dependency_overrides[get_rag_service] = mock_rag_service

def test_websocket_receives_mocked_response():
    client = TestClient(app)
    with client.websocket_connect("/api/v1/ws/test_client") as websocket:
        websocket.send_text("Hello")
        data = websocket.receive_text()
        assert data == "Mocked RAG response"
```

## 3. Integration Testing

**Goal:** Verify the interaction between key components of the system.

**Tools:** `pytest`, `httpx.AsyncClient`, Dockerized services (Redis)

**Example:** Send a message via a WebSocket, then connect directly to the test Redis instance to confirm that the `ContextManagerService` correctly saved the message.

```python
# tests/integration/test_chat_integration.py
import pytest
import aioredis
from fastapi.testclient import TestClient
from src.app.main import app

@pytest.mark.asyncio
async def test_websocket_saves_to_redis():
    # Arrange
    client = TestClient(app)
    redis = await aioredis.from_url("redis://localhost:6379") # Connect to test Redis

    # Act
    with client.websocket_connect("/api/v1/ws/integration_test") as websocket:
        websocket.send_text("Integration test message")
        # Allow time for processing
        _ = websocket.receive_text()

    # Assert
    history_json = await redis.get("integration_test")
    history = json.loads(history_json)
    assert history[-1]["role"] == "user"
    assert history[-1]["content"] == "Integration test message"

    # Cleanup
    await redis.delete("integration_test")
```

## 4. Testing Business Logic (Context/Precision)

**Goal:** Verify that the application correctly implements specific business rules.

**Tools:** `pytest`, `httpx.AsyncClient`, Mocking

**Example:** Create a test case that sends four consecutive low-relevance questions and verifies that on the fourth question:
A. The context is cleared.
B. The `RAGService` is called with `low_precision=True`.
C. The user receives a warning message.

```python
# tests/logic/test_clarification_loop.py
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from src.app.main import app
from src.app.dependencies.rag import get_rag_service
from src.app.dependencies.context import get_context_manager

# Mock RAG to return low score
async def mock_low_score_rag():
    class MockRAGService:
        async def aquery(self, query: str, low_precision: bool = False):
            if low_precision:
                return "Low precision answer", "http://mock.link", 0.5
            return "Needs clarification", "http://mock.link", 0.5
    yield MockRAGService()

app.dependency_overrides[get_rag_service] = mock_low_score_rag

def test_clarification_loop_and_reset():
    client = TestClient(app)
    with client.websocket_connect("/api/v1/ws/logic_test") as websocket:
        # 1st, 2nd, 3rd messages: ask for clarification
        for _ in range(3):
            websocket.send_text("Irrelevant question")
            response = websocket.receive_text()
            assert "Не могли бы вы уточнить ваш вопрос?" in response

        # 4th message: should trigger low-precision answer and context reset
        websocket.send_text("Fourth irrelevant question")
        response = websocket.receive_text()

        # Assert
        assert "Точность ответа может быть низкой" in response
        assert "Low precision answer" in response
```

## 5. WebSocket Load Testing

**Goal:** Verify the stability and performance of the application under high concurrent load.

**Tools:** `pytest` with a load testing library (e.g., a custom script using `asyncio` and `websockets` library).

**Example:** Create a test that opens 100 simultaneous WebSocket connections and sends messages concurrently, checking for errors.

```python
# tests/load/test_websocket_concurrency.py
import pytest
import asyncio
import websockets

async def client_task(client_id):
    uri = f"ws://localhost:8000/api/v1/ws/{client_id}"
    try:
        async with websockets.connect(uri) as websocket:
            for i in range(10):
                await websocket.send(f"Message {i} from client {client_id}")
                response = await websocket.recv()
                assert response is not None
        return True
    except Exception as e:
        print(f"Client {client_id} failed: {e}")
        return False

@pytest.mark.asyncio
async def test_high_concurrency():
    num_clients = 100
    tasks = [client_task(f"load_test_{i}") for i in range(num_clients)]
    results = await asyncio.gather(*tasks)
    assert all(results), "Not all WebSocket clients completed successfully."

```
