from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')
# Variables check
if DB_URI is None:
    raise ValueError("Переменные не найдены. Проверьте файл .env.")

engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
