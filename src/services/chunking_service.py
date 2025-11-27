import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import tiktoken
import nltk
from uuid import uuid4

# Ensure nltk data is available (can be moved to a setup script)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from src.app.logging_config import get_logger

logger = get_logger(__name__)

class ChunkingService:
    """
    Service for splitting text into chunks with advanced logic:
    - Recursive splitting (hierarchical).
    - Token-based splitting (tiktoken).
    - Sentence boundary detection (nltk).
    - Contextual headers (Contextual Retrieval).
    - Table preservation.
    - Inter-page overlap preservation.
    """

    def __init__(self, model_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_name)

    def create_chunks_with_metadata(
        self, 
        parsed_pages: List[Dict[str, Any]], 
        document_id: int,
        document_title: str,
        chunk_size: int = 1000,  # Увеличен размер чанка для лучшего контекста
        overlap: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Creates chunks from parsed pages with rich metadata and contextual headers.
        """
        all_chunks_data = []
        previous_overlap_sentences = []

        for page in parsed_pages:
            page_content = page.get('content', '')
            page_num = page.get('page_number')
            content_type = page.get('type', 'text')
            sheet_name = page.get('sheet_name')

            # Очистка текста: убираем разрывы строк и исправляем переносы
            page_content = self._clean_text(page_content)

            # Contextual Header for this page/section
            # Example: "Документ: Инструкция. Стр: 5"
            context_header = f"Документ: {document_title}. "
            if sheet_name:
                context_header += f"Лист: {sheet_name}. "
            elif page_num:
                context_header += f"Стр: {page_num}. "
            
            # Special handling for tables: try to keep them as single chunk if possible
            if content_type == 'table':
                tokens = self._count_tokens(page_content)
                if tokens <= chunk_size:
                    # Table fits in one chunk!
                    chunk_text = f"{context_header}\n{page_content}"
                    all_chunks_data.append(self._create_chunk_dict(
                        document_id, chunk_text, page_num, 0, content_type, sheet_name
                    ))
                    # Reset overlap because table breaks the flow usually
                    previous_overlap_sentences = [] 
                    continue

            # Recursive splitting for text or large tables
            chunks, current_tail = self._split_with_recursive(
                page_content, 
                chunk_size=chunk_size, 
                overlap=overlap, 
                previous_overlap_sentences=previous_overlap_sentences,
                context_header=context_header
            )
            
            previous_overlap_sentences = current_tail

            for i, chunk_text in enumerate(chunks):
                all_chunks_data.append(self._create_chunk_dict(
                    document_id, chunk_text, page_num, i, content_type, sheet_name
                ))

        return all_chunks_data

    def _clean_text(self, text: str) -> str:
        """
        Очищает текст от артефактов форматирования:
        1. Убирает переносы слов (например, "каче- ство" -> "качество").
        2. Заменяет одиночные переводы строк на пробелы (reflow).
        3. Сохраняет абзацы (двойные переводы строк).
        """
        if not text:
            return text
        
        # 1. Исправляем переносы: "слово- продолжение" или "слово-\nпродолжение" -> "словопродолжение"
        # Паттерн: дефис + любые пробелы/переводы строк
        text = re.sub(r'-[\s\n]+', '', text)
        
        # 2. Заменяем одиночные переводы строк на пробелы (для слияния разорванных предложений)
        # Сохраняем абзацы (двойные и более переводы строк)
        # Сначала заменяем все последовательности из 2+ переводов строк на маркер
        text = re.sub(r'\n{2,}', '<<<PARAGRAPH>>>', text)
        # Теперь заменяем оставшиеся одиночные переводы строк на пробелы
        text = re.sub(r'\n', ' ', text)
        # Восстанавливаем абзацы
        text = re.sub(r'<<<PARAGRAPH>>>', '\n\n', text)
        
        # 3. Убираем множественные пробелы
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

    def _split_with_recursive(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        previous_overlap_sentences: List[str],
        context_header: str = ""
    ) -> Tuple[List[str], List[str]]:
        """
        Recursive splitting with overlap preservation between pages.
        """
        if not text:
            return [], previous_overlap_sentences

        # Hierarchy of separators (from large to small) insted of paragraphs
        separators = [
            "\n\n\n",  # Triple newline (sections)
            "\n\n",    # Double newline (paragraphs)
            "\n",      # Single newline (lines)
        ]

        # 1. Recursively split text
        text_parts = self._recursive_split(text, separators)
        
        # 2. Split each part into sentences
        all_sentences = []
        for part in text_parts:
            sentences = self._split_into_sentences(part)
            all_sentences.extend(sentences)

        # 3. Add overlap from previous page
        if previous_overlap_sentences:
            all_sentences = previous_overlap_sentences + all_sentences

        # 4. Form chunks considering token limits
        chunks, tail_sentences = self._form_chunks_from_sentences(
            all_sentences,
            chunk_size,
            overlap,
            context_header
        )

        return chunks, tail_sentences

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using a hierarchy of separators."""
        if not separators:
            return [text]
        
        current_sep = separators[0]
        remaining_seps = separators[1:]
        
        if current_sep not in text:
            # Separator not found, try the next one
            return self._recursive_split(text, remaining_seps)
        
        # Split by current separator
        parts = text.split(current_sep)
        result = []
        
        for part in parts:
            if not part.strip():
                continue
            # Recursively process each part
            sub_parts = self._recursive_split(part, remaining_seps)
            result.extend(sub_parts)
        
        return result if result else [text]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences with smart handling of numbered lists."""
        try:
            sentences = nltk.sent_tokenize(text, language="russian")
        except Exception:
            sentences = [text]

        # Merge short items (e.g., "1.") with the next sentence
        adjusted_sentences = []
        i = 0
        while i < len(sentences):
            current_sentence = sentences[i]
            is_number_only = re.match(r'^\s*\d+\.?\s*$', current_sentence)
            is_too_short = len(current_sentence.split()) <= 2
            
            if (is_number_only or is_too_short) and (i + 1 < len(sentences)):
                sentences[i + 1] = current_sentence.strip() + ' ' + sentences[i + 1].strip()
            else:
                adjusted_sentences.append(current_sentence.strip())
            i += 1
        
        return adjusted_sentences

    def _form_chunks_from_sentences(
        self,
        sentences: List[str],
        chunk_size: int,
        overlap: int,
        context_header: str
    ) -> Tuple[List[str], List[str]]:
        """
        Forms chunks from sentences considering tokens and overlap.
        Returns (chunks, tail_sentences) where tail is for overlap with the next page.
        """
        if not sentences:
            return [], []

        chunks = []
        current_sentences_in_chunk = []
        current_chunk_text_body = ""
        
        # Account for header tokens
        header_tokens = self._count_tokens(context_header) if context_header else 0
        effective_chunk_size = chunk_size - header_tokens
        
        if effective_chunk_size < 100:
            logger.warning("Context header is too large relative to chunk size!")
            effective_chunk_size = chunk_size

        for sentence in sentences:
            current_sentences_in_chunk.append(sentence)
            current_chunk_text_body += ' ' + sentence
            
            # Check size
            if self._count_tokens(current_chunk_text_body) >= effective_chunk_size:
                # Finalize chunk
                full_chunk_text = f"{context_header}\n{current_chunk_text_body.strip()}" if context_header else current_chunk_text_body.strip()
                chunks.append(full_chunk_text)

                # Calculate overlap for NEXT chunk
                overlap_sentences = []
                overlap_tokens = 0
                
                for s in reversed(current_sentences_in_chunk):
                    s_tokens = self._count_tokens(s)
                    overlap_tokens += s_tokens
                    overlap_sentences.insert(0, s)
                    if overlap_tokens >= overlap:
                        break
                
                # Start new chunk with overlap
                current_sentences_in_chunk = overlap_sentences
                current_chunk_text_body = " ".join(overlap_sentences)

        # Finalize last chunk
        if current_chunk_text_body.strip():
            last_chunk_body = current_chunk_text_body.strip()
            full_last_chunk = f"{context_header}\n{last_chunk_body}" if context_header else last_chunk_body
            
            if not chunks or chunks[-1] != full_last_chunk:
                chunks.append(full_last_chunk)

        # Calculate tail for the NEXT PAGE (CRITICAL!)
        tail_sentences = []
        tail_tokens = 0
        for s in reversed(current_sentences_in_chunk):
            s_tokens = self._count_tokens(s)
            if tail_tokens + s_tokens <= overlap:
                tail_sentences.insert(0, s)
                tail_tokens += s_tokens
            else:
                break

        return chunks, tail_sentences

    def _create_chunk_dict(self, doc_id, content, page_num, chunk_index, c_type, sheet_name):
        return {
            'document_id': doc_id,
            'content': content,
            'page_number': page_num,
            'chunk_index': chunk_index,
            'type': c_type,
            'sheet_name': sheet_name,
            'qdrant_id': str(uuid4())
        }

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

chunking_service = ChunkingService()
