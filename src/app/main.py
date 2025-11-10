from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routers import chat

app = FastAPI(title="RAG Chatbot")

templates = Jinja2Templates(directory="src/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.include_router(chat.router, prefix="/api/v1")
