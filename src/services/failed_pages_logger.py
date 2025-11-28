"""
Логгер для отслеживания проблемных страниц при парсинге PDF документов.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class FailedPagesLogger:
    """
    Логгер для сохранения информации о нераспознанных страницах.
    Использует JSON Lines формат для компактного хранения.
    """
    
    def __init__(self, log_file: str = "failed_pages.jsonl"):
        """
        Args:
            log_file: Путь к файлу лога (по умолчанию failed_pages.jsonl)
        """
        self.log_file = Path(log_file)
        # Создаём директорию если не существует
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_failed_page(
        self, 
        document_name: str, 
        page_number: int, 
        reason: str, 
        processing_time: Optional[float] = None
    ):
        """
        Добавляет запись о неудачной странице в лог.
        
        Args:
            document_name: Название документа
            page_number: Номер страницы (1-indexed)
            reason: Причина ошибки
            processing_time: Время обработки в секундах (опционально)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "document": document_name,
            "page": page_number,
            "reason": reason,
        }
        
        if processing_time is not None:
            entry["processing_time_seconds"] = round(processing_time, 2)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def get_failed_pages_by_document(self, document_name: str) -> list:
        """
        Получает список проблемных страниц для указанного документа.
        
        Args:
            document_name: Название документа
            
        Returns:
            Список записей о проблемных страницах
        """
        failed_pages = []
        
        if not self.log_file.exists():
            return failed_pages
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get('document') == document_name:
                            failed_pages.append(entry)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return failed_pages
    
    def get_all_failed_pages(self) -> list:
        """
        Получает все записи о проблемных страницах.
        
        Returns:
            Список всех записей
        """
        all_failed = []
        
        if not self.log_file.exists():
            return all_failed
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_failed.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return all_failed
    
    def get_statistics(self) -> dict:
        """
        Получает статистику по проблемным страницам.
        
        Returns:
            Словарь со статистикой
        """
        all_failed = self.get_all_failed_pages()
        
        if not all_failed:
            return {
                "total_failed_pages": 0,
                "unique_documents": 0,
                "most_problematic_documents": []
            }
        
        # Подсчёт страниц по документам
        doc_counts = {}
        for entry in all_failed:
            doc = entry.get('document', 'unknown')
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
        
        # Сортировка документов по количеству проблемных страниц
        most_problematic = sorted(
            doc_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            "total_failed_pages": len(all_failed),
            "unique_documents": len(doc_counts),
            "most_problematic_documents": [
                {"document": doc, "failed_pages_count": count}
                for doc, count in most_problematic
            ]
        }


# Singleton instance
failed_pages_logger = FailedPagesLogger()
