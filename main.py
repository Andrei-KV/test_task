from sqlalchemy import create_engine, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from datetime import datetime
from uuid import UUID
import re

# -----------
# Подключение к БД и создание таблиц
# --------------------

DB_URI = "postgresql+psycopg2://postgres:1765362@localhost:5432/legal_rag_db"
engine = create_engine(DB_URI)

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    link_to_file: Mapped[str] = mapped_column(String)
    load_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.document_id"))
    content: Mapped[str] = mapped_column(Text)
    # Для связки с векторной БД
    qdrant_id: Mapped[str] = mapped_column(String(255))
    # Обратная связь
    document: Mapped["Document"] = relationship(back_populates="chunks")

# Применение изменений к базе данных
Base.metadata.create_all(engine)

# ----------------------
# Чтение файла
# -------------------
try:
    with open('codex2.md', 'r', encoding='utf-8') as f:
        raw_content = f.read()
except FileNotFoundError:
    print("Ошибка: Файл codex2.md не найден.")
    raw_content = ""

# ---------------------
# Сохранение данных файла в таблицу
# ------------------------

with Session(engine) as session:
    
    # Создаем запись в таблице (Document)
    new_document = Document(
        title="Codex 2 - Тестовый Юридический Документ",
        # Ссылка на файл, где он хранится
        link_to_file="codex2.md" 
        # load_date автоматически устанавливается
    )
    session.add(new_document)
    session.commit()

# ??? После commit() Python-объект new_document в памяти не "знает", какой ID ему присвоил PostgreSQL
# Команда session.refresh(new_document) заставляет $\text{SQLAlchemy}$ 
# выполнить быстрый запрос SELECT к базе данных, чтобы обновить объект new_document в памяти Python, 
# заполнив его всеми полями, включая только что сгенерированный document_id.
# !! УТОЧНИТЬ надо ли
    session.refresh(new_document)
    document_id = new_document.document_id


#---------------------------------
# Очистка (!!Очистку для разных типов файлов)
# -----------------------------------
# 1. Удаление HTML/Markdown артефактов
# Паттерн 1: Удаляет HTML-теги (<...> и </...>)
cleaned_content = re.sub(r'<[^>]+>', ' ', raw_content)

# Паттерн 2: Удаляет HTML-сущности (например, &nbsp;, &lt;, &#x27;)
cleaned_content = re.sub(r'&[a-z]+;|&#x?[0-9a-f]+;', ' ', cleaned_content)

# 2. Нормализация пробелов
# Паттерн: Заменяет любые последовательности пробельных символов (пробел, \n, \t) на один пробел.
# .strip() удаляет пробел с начала и конца
cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()

print(f"Исходная длина: {len(raw_content)} символов")
print(f"Очищенная длина: {len(cleaned_content)} символов")
print(cleaned_content[:500]) # Вывод первых 500 символов для проверки


# -----------------------------
# Разбиение на chunks, векторизация файла и добавление в векторную БД
# -----------------------------

