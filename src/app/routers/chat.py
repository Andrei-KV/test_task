from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from ..dependencies.websocket_manager import manager
from ..dependencies.rag import get_rag_service
from ..dependencies.context import get_context_manager
from ...services.rag_service import RAGService
from ...services.context_manager import ContextManagerService
from src.app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    rag_service: RAGService = Depends(get_rag_service),
    context_manager: ContextManagerService = Depends(get_context_manager),
):
    await manager.connect(websocket)
    # Send history to user on connect
    history = await context_manager.get_full_history(client_id)
    for message in history:
        await manager.send_personal_message(text=f"{message['role']}: {message['content']}", websocket=websocket)


    try:
        while True:
            data = await websocket.receive_text()
            await context_manager.add_message(client_id, "user", data)
            
            # Send "Processing..." message
            await manager.send_personal_message(text="Идёт обработка...", websocket=websocket)

            context_window = await context_manager.get_context_window(client_id)
            context_query = " ".join([msg["content"] for msg in context_window])

            answer, web_link, score, title = await rag_service.aquery(context_query)
            logger.info(f'answer: {answer} \nweb_link: {web_link} \nscore: {score}')

            if score < 0.7:
                clarification_count = await context_manager.get_clarification_count(client_id)
                if clarification_count < 2:
                    await context_manager.increment_clarification_count(client_id)
                    clarification_question = "Не могли бы вы уточнить вопрос?"
                    await context_manager.add_message(client_id, "bot", clarification_question)
                    await manager.send_personal_message(text=clarification_question, websocket=websocket)
                else:
                    await context_manager.reset_context(client_id)
                    answer, web_link, score, title = await rag_service.aquery(data, low_precision=True)
                    warning = "Точность ответа может быть низкой. Попробуйте переформулировать вопрос."
                    final_answer = f"{warning}\n\n{answer}"
                    await context_manager.add_message(client_id, "bot", final_answer)
                    await manager.send_personal_message(text=final_answer, websocket=websocket, web_link=web_link, title=title)
                    #  Reset context after providing a low-precision answer
                    await context_manager.reset_context(client_id)
            else:
                await context_manager.add_message(client_id, "bot", answer)
                await manager.send_personal_message(text=answer, websocket=websocket, web_link=web_link, title=title)
                # Reset context after a successful answer
                await context_manager.reset_context(client_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
