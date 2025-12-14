from sqlalchemy import (
    String, Text, DateTime, ForeignKey, UniqueConstraint, Integer
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import datetime

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint('title', 'drive_file_id', name='uq_title_id'),
    )

    document_id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    drive_file_id: Mapped[str] = mapped_column(String(255), unique=True)
    web_link: Mapped[str] = mapped_column(String)
    load_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

import uuid

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.document_id"))
    document_title: Mapped[str] = mapped_column(String(255))  # Название документа
    page_number: Mapped[int] = mapped_column(Integer, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer)  # Глобальный индекс чанка в документе
    content: Mapped[str] = mapped_column(Text)  # Чистый текст без метаданных
    content_type: Mapped[str] = mapped_column(String(50), default='text')  # 'text', 'ocr_text', 'table', 'image'
    sheet_name: Mapped[str] = mapped_column(String(255), nullable=True)  # Для Excel файлов
    qdrant_id: Mapped[str] = mapped_column(String(255), unique=True)  # ID для векторной БД
    document: Mapped["Document"] = relationship(back_populates="chunks")

