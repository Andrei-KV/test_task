from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.rag_pipeline import run_rag_pipeline
import logging
import aioredis
from config import REDIS_HOST, REDIS_PORT, REDIS_DB
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Pipeline API",
    description="API for interacting with the RAG pipeline.",
    version="1.0.0"
)

redis_client = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}", encoding="utf-8", decode_responses=True)

class Query(BaseModel):
    user_id: str
    text: str

async def get_conversation_history(user_id: str) -> list:
    history_json = await redis_client.get(f"history:{user_id}")
    if history_json:
        return json.loads(history_json)
    return []

async def add_to_conversation_history(user_id: str, user_query: str, bot_response: str):
    history = await get_conversation_history(user_id)
    history.append({"user": user_query, "bot": bot_response})
    if len(history) > 2:
        history = history[-2:]
    await redis_client.set(f"history:{user_id}", json.dumps(history))

async def get_clarification_count(user_id: str) -> int:
    count = await redis_client.get(f"clarify_count:{user_id}")
    return int(count) if count else 0

async def increment_clarification_count(user_id: str):
    count = await get_clarification_count(user_id)
    await redis_client.set(f"clarify_count:{user_id}", count + 1)

async def reset_clarification_count(user_id: str):
    await redis_client.delete(f"clarify_count:{user_id}")

@app.get("/")
async def read_root():
    return {"message": "RAG API is running"}

@app.post("/query")
async def query(query: Query):
    logger.info(f"Received query from user {query.user_id}: {query.text}")
    try:
        history = await get_conversation_history(query.user_id)
        final_answer, web_link, score = await run_rag_pipeline(query.text, history)

        if score < 0.7:
            clarification_count = await get_clarification_count(query.user_id)
            if clarification_count < 3:
                await increment_clarification_count(query.user_id)
                return {"answer": "Уточните, пожалуйста, ваш вопрос.", "status": "clarification_needed"}
            else:
                await reset_clarification_count(query.user_id)
                await redis_client.delete(f"history:{query.user_id}")
                final_answer, web_link, score = await run_rag_pipeline(query.text)
                return {
                    "answer": f"Точность ответа может быть низкой (оценка: {score:.2f}). {final_answer}",
                    "source_link": web_link,
                    "status": "low_confidence"
                }
        else:
            await reset_clarification_count(query.user_id)
            await add_to_conversation_history(query.user_id, query.text, final_answer)
            response = {
                "answer": final_answer,
                "source_link": web_link,
                "status": "success"
            }
            logger.info(f"Successfully processed query for user {query.user_id}: {query.text}")
            return response

    except Exception as e:
        logger.error(f"Error processing query for user {query.user_id} '{query.text}': {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the request.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
