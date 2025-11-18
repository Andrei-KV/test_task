from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from src.database.models import Base

import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')
# Variables check
if DB_URI is None:
    raise ValueError("Переменные не найдены. Проверьте файл .env.")

# Synchronous engine for init_db
# engine = create_engine(DB_URI)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Asynchronous engine for the application
async_engine = create_async_engine(DB_URI)
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)