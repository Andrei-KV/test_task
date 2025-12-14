from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from ..dependencies.websocket_manager import manager
from ..dependencies.rag import get_rag_service
from ..dependencies.context import get_context_manager
from ...services.rag_service_opensearch import RAGService
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
        # Pass document metadata if available to restore links
        docs = message.get("documents")
        await manager.send_personal_message(
            text=f"{message['role']}: {message['content']}",
            websocket=websocket,
            documents=docs
        )


    try:
        while True:
            data = await websocket.receive_text()
            await context_manager.add_message(client_id, "user", data)
            
            # Send "Processing..." message with loading indicator
            await manager.send_personal_message(text="Идёт обработка...", websocket=websocket, is_loading=True)

            history = await context_manager.get_full_history(client_id)
            context_query = ""

            # The most recent message is the user's current input, which is at history[-1]
            is_clarification_response = False
            if len(history) >= 2:
                if history[-2].get("role") == "bot" and "уточнить вопрос" in history[-2].get("content", ""):
                    is_clarification_response = True
            
            if is_clarification_response:
                # Find the user's original question (before the bot's clarification request)
                original_question = ""
                for message in reversed(history[:-2]):
                    if message.get("role") == "user":
                        original_question = message.get("content")
                        break
                
                if original_question:
                    # Combine original question with the user's new clarifying answer, excluding the bot's message
                    context_query = f"{original_question} {data}"
                else:
                    # Fallback to just the current message if the original couldn't be found
                    context_query = data
            else:
                # Default context window for new questions
                # context_window = await context_manager.get_context_window(client_id)
                # context_query = " ".join([msg["content"] for msg in context_window])

                context_query = data
            
            logger.info(f'Context query: {context_query}')
            answer, documents_info, score = await rag_service.aquery(context_query)
            
            logger.info(f'answer: {answer} \ndocuments: {len(documents_info)} \nscore: {score}')

            if score < 7.4:
                clarification_count = await context_manager.get_clarification_count(client_id)
                if clarification_count < 1:
                    await context_manager.increment_clarification_count(client_id)
                    clarification_question = "Не могли бы вы уточнить вопрос?"
                    await context_manager.add_message(client_id, "bot", clarification_question)
                    await manager.send_personal_message(text=clarification_question, websocket=websocket)
                else:
                    # await context_manager.reset_context(client_id)
                    answer, documents_info, score = await rag_service.aquery(context_query, low_precision=True)
                    
                    warning = "Точность ответа может быть низкой. Попробуйте переформулировать вопрос."
                    final_answer = f"{warning}\n\n{answer}"
                    # 6. Save context
                    await context_manager.add_message(client_id, "user", data)
                    await context_manager.add_message(client_id, "bot", final_answer, documents=documents_info)
                    await manager.send_personal_message(text=final_answer, websocket=websocket, documents=documents_info)
                    #  Reset context after providing a low-precision answer
                    # await context_manager.reset_context(client_id)
            else:
                final_answer = answer
                # 6. Save context
                await context_manager.add_message(client_id, "user", data)
                await context_manager.add_message(client_id, "bot", final_answer, documents=documents_info)
                await manager.send_personal_message(text=final_answer, websocket=websocket, documents=documents_info)
                # Reset context after a successful answer
                # await context_manager.reset_context(client_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
