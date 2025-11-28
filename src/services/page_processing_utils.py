"""
Утилиты для обработки страниц PDF с поддержкой таймаутов и multiprocessing.
"""

import signal
from contextlib import contextmanager
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Исключение при превышении таймаута"""
    pass


@contextmanager
def timeout(seconds: int):
    """
    Контекстный менеджер для установки таймаута на операцию.
    
    Args:
        seconds: Максимальное время выполнения в секундах
        
    Raises:
        TimeoutError: Если операция превысила лимит времени
        
    Example:
        with timeout(120):
            # код, который должен выполниться за 2 минуты
            process_page()
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Сохраняем оригинальный обработчик
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    
    # Устанавливаем alarm
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Отменяем alarm и восстанавливаем обработчик
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def process_single_page_with_timeout(
    file_path: str,
    page_num: int,
    timeout_seconds: int = 120
) -> Tuple[int, bool, str]:
    """
    Обработка одной страницы PDF с OCR и таймаутом.
    
    Args:
        file_path: Путь к PDF файлу
        page_num: Номер страницы (0-indexed)
        timeout_seconds: Таймаут в секундах (по умолчанию 120)
        
    Returns:
        Tuple[page_num, success, content/error_message]
        
    Example:
        page_num, success, result = process_single_page_with_timeout(
            "document.pdf", 
            2,  # 3-я страница
            120
        )
        if success:
            print(f"Page {page_num + 1}: {result}")
        else:
            print(f"Failed: {result}")
    """
    try:
        from pdf2image import convert_from_path
        from .document_parser import DocumentParser
        
        with timeout(timeout_seconds):
            # Конвертируем страницу в изображение
            images = convert_from_path(
                file_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=70,
                fmt='jpeg',
                grayscale=True
            )
            
            if not images:
                return (page_num, False, "No image generated from PDF")
            
            # Выполняем OCR
            parser = DocumentParser()
            page_text = parser._extract_text_with_tesseract(images[0])
            
            # Очистка
            images[0].close()
            del images
            
            return (page_num, True, page_text)
            
    except TimeoutError as e:
        logger.error(f"Timeout processing page {page_num + 1}: {e}")
        return (page_num, False, f"Timeout: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1}: {e}")
        return (page_num, False, f"Error: {str(e)}")
