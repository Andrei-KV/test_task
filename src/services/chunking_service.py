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
from ..config import (
    CHUNKING_SIZE,
    CHUNKING_OVERLAP,
    CHUNKING_OVERLAP_LIMIT_RATIO,
    CHUNKING_SENTENCE_HARD_LIMIT
)

logger = get_logger(__name__)

class ChunkingService:
    """
    Сервис для разбиения текста на чанки.
    
    КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Метаданные НЕ добавляются в текст чанка.
    Они возвращаются как отдельные поля в структуре чанка.
    
    Особенности:
    - Рекурсивное разбиение (иерархическое).
    - Разбиение на основе токенов (tiktoken).
    - Определение границ предложений (nltk).
    - Сохранение таблиц.
    - Сохранение overlap между страницами.
    """

    def __init__(self, model_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_name)

    def create_chunks_with_metadata(
        self, 
        parsed_pages: List[Dict[str, Any]], 
        document_id: int,
        document_title: str,
        chunk_size: int = CHUNKING_SIZE,
        overlap: int = CHUNKING_OVERLAP
    ) -> List[Dict[str, Any]]:
        """
        Создает чанки из распаршенных страниц с метаданными.
        
        Метаданные (document_title, page_number, content_type и т.д.) 
        хранятся в отдельных полях, а НЕ добавляются в content.
        """
        all_chunks_data = []
        previous_overlap_sentences = []
        global_chunk_index = 0

        for page in parsed_pages:
            page_content = page.get('content', '')
            page_num = page.get('page_number')
            content_type = page.get('type', 'text')
            sheet_name = page.get('sheet_name')

            # Очистка текста
            page_content = self._clean_text(page_content)
            
            if not page_content.strip():
                continue

            # Специальная обработка таблиц
            if content_type == 'table':
                tokens = self._count_tokens(page_content)
                if tokens <= chunk_size:
                    all_chunks_data.append(self._create_chunk_dict(
                        document_id=document_id,
                        document_title=document_title,
                        content=page_content,
                        page_num=page_num,
                        chunk_index=global_chunk_index,
                        content_type=content_type,
                        sheet_name=sheet_name
                    ))
                    global_chunk_index += 1
                    previous_overlap_sentences = []
                    continue

            # Рекурсивное разбиение
            chunks, current_tail = self._split_with_recursive(
                page_content, 
                chunk_size=chunk_size, 
                overlap=overlap, 
                previous_overlap_sentences=previous_overlap_sentences
            )
            
            previous_overlap_sentences = current_tail

            for chunk_text in chunks:
                all_chunks_data.append(self._create_chunk_dict(
                    document_id=document_id,
                    document_title=document_title,
                    content=chunk_text,
                    page_num=page_num,
                    chunk_index=global_chunk_index,
                    content_type=content_type,
                    sheet_name=sheet_name
                ))
                global_chunk_index += 1

        return all_chunks_data

    def _clean_text(self, text: str) -> str:
        """
        Очищает текст от артефактов форматирования.
        """
        if not text:
            return text
        
        # Исправляем переносы слов
        text = re.sub(r'-[\s\n]+', '', text)
        
        # Сохраняем абзацы, но убираем одиночные переводы строк
        text = re.sub(r'\n{2,}', '<<<PARAGRAPH>>>', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'<<<PARAGRAPH>>>', '\n\n', text)
        
        # Убираем множественные пробелы
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

    def _split_with_recursive(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        previous_overlap_sentences: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Рекурсивное разбиение с сохранением overlap между страницами.
        """
        if not text:
            return [], previous_overlap_sentences

        separators = ["\n\n\n", "\n\n", "\n"]
        text_parts = self._recursive_split(text, separators)
        
        all_sentences = []
        for part in text_parts:
            sentences = self._split_into_sentences(part)
            all_sentences.extend(sentences)

        if previous_overlap_sentences:
            all_sentences = previous_overlap_sentences + all_sentences

        chunks, tail_sentences = self._form_chunks_from_sentences(
            all_sentences,
            chunk_size,
            overlap
        )

        return chunks, tail_sentences

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Рекурсивно разбивает текст по иерархии разделителей."""
        if not separators:
            return [text]
        
        current_sep = separators[0]
        remaining_seps = separators[1:]
        
        if current_sep not in text:
            return self._recursive_split(text, remaining_seps)
        
        parts = text.split(current_sep)
        result = []
        
        for part in parts:
            if not part.strip():
                continue
            sub_parts = self._recursive_split(part, remaining_seps)
            result.extend(sub_parts)
        
        return result if result else [text]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        try:
            sentences = nltk.sent_tokenize(text, language="russian")
        except Exception:
            sentences = [text]

        # Объединяем короткие элементы с следующим предложением
        adjusted_sentences = []
        i = 0
        while i < len(sentences):
            current_sentence = sentences[i]
            is_number_only = re.match(r'^\s*\d+\.?\s*$', current_sentence)
            is_too_short = len(current_sentence.split()) <= 2
            
            if (is_number_only or is_too_short) and (i + 1 < len(sentences)):
                sentences[i + 1] = current_sentence.strip() + ' ' + sentences[i + 1].strip() # Fixed space issue
            else:
                adjusted_sentences.append(current_sentence.strip())
            i += 1
        
        return adjusted_sentences

    def _form_chunks_from_sentences(
        self,
        sentences: List[str],
        chunk_size: int,
        overlap: int
    ) -> Tuple[List[str], List[str]]:
        """
        Формирует чанки из предложений с учетом токенов и overlap.
        """
        if not sentences:
            return [], []

        chunks = []
        current_sentences_in_chunk = []
        current_chunk_text = ""

        # Strict Overlap Logic: 
        # Не допускаем превышения overlap более чем на 30% (overlap * 1.3),
        # если только само первое предложение не является огромным.
        OVERLAP_LIMIT = overlap * CHUNKING_OVERLAP_LIMIT_RATIO

        for sentence in sentences:
            current_sentences_in_chunk.append(sentence)
            current_chunk_text += ' ' + sentence
            
            if self._count_tokens(current_chunk_text) >= chunk_size:
                # Финализируем чанк
                chunk_text = current_chunk_text.strip()
                chunks.append(chunk_text)

                # Вычисляем overlap для следующего чанка
                overlap_sentences = []
                overlap_tokens = 0
                
                for s in reversed(current_sentences_in_chunk):
                    s_tokens = self._count_tokens(s)
                    
                    # Если у нас уже есть контент в overlap, и добавление следующего 
                    # предложения превысит жесткий лимит - стоп.
                    if overlap_tokens > 0 and (overlap_tokens + s_tokens > OVERLAP_LIMIT):
                        break
                    
                    overlap_tokens += s_tokens
                    overlap_sentences.insert(0, s)
                    
                    if overlap_tokens >= overlap:
                        break
                
                # Защита от бесконечного цикла (когда весь чанк уходит в overlap)
                if len(overlap_sentences) == len(current_sentences_in_chunk) and len(overlap_sentences) > 0:
                     overlap_sentences.pop(0)

                current_sentences_in_chunk = overlap_sentences
                current_chunk_text = " ".join(overlap_sentences)

        # Финализируем последний чанк
        if current_chunk_text.strip():
            last_chunk = current_chunk_text.strip()
            if not chunks or chunks[-1] != last_chunk:
                chunks.append(last_chunk)

        # Tail для следующей страницы (аналогичная логика)
        tail_sentences = []
        tail_tokens = 0
        for s in reversed(current_sentences_in_chunk):
            s_tokens = self._count_tokens(s)
            
            # То же правило: не превышать лимит слишком сильно
            if tail_tokens > 0 and (tail_tokens + s_tokens > OVERLAP_LIMIT):
                break
                
            if tail_tokens + s_tokens <= OVERLAP_LIMIT: # Используем чуть более мягкий лимит тут тоже
                 tail_sentences.insert(0, s)
                 tail_tokens += s_tokens
            else:
                 # Если одно предложение больше лимита, берем его (если пустой tail) и выходим
                 if tail_tokens == 0:
                     tail_sentences.insert(0, s)
                 break

        return chunks, tail_sentences

    def _create_chunk_dict(
        self, 
        document_id: int,
        document_title: str,
        content: str, 
        page_num: Optional[int], 
        chunk_index: int, 
        content_type: str, 
        sheet_name: Optional[str]
    ) -> Dict[str, Any]:
        """
        Создает структуру чанка с метаданными.
        
        ВАЖНО: Метаданные хранятся в отдельных полях, а НЕ в content.
        """
        return {
            'document_id': document_id,
            'document_title': document_title,
            'content': content, 
            'page_number': page_num,
            'chunk_index': chunk_index,
            'content_type': content_type,
            'sheet_name': sheet_name,
            'qdrant_id': str(uuid4())
        }
    
    def _count_tokens(self, text: str) -> int:
        """Считает количество токенов в тексте."""
        return len(self.tokenizer.encode(text))


# Singleton instance
chunking_service = ChunkingService()
