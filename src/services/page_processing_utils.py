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
