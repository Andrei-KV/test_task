from sqlalchemy import (
    String, Text, DateTime, ForeignKey, UniqueConstraint
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
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.document_id"))
    content: Mapped[str] = mapped_column(Text)
    qdrant_id: Mapped[str] = mapped_column(String(255))
    document: Mapped["Document"] = relationship(back_populates="chunks")
