import os
import logging
import tempfile
import mimetypes
import io
import re
import time
from typing import List, Dict, Any, Tuple

# Библиотеки для обработки форматов
import pypandoc
import fitz  # PyMuPDF
import pandas as pd
import markdown
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Наши утилиты
from .page_processing_utils import timeout, TimeoutError
from .failed_pages_logger import failed_pages_logger

# Настройка логирования
from src.app.logging_config import get_logger
logger = get_logger(__name__)

# Ограничиваем количество потоков Tesseract для предотвращения перегрузки CPU
os.environ['OMP_THREAD_LIMIT'] = '1'

# Настройки адаптивного OCR
ADAPTIVE_OCR_DPI = 250
ADAPTIVE_SIZE_THRESHOLDS = {
    'A2': {'max_size': 4500, 'max_dimension': 1700},  # Большие страницы
    'A3': {'max_size': 3500, 'max_dimension': 1500},  # Средние страницы
    'A4': {'max_size': 2500, 'max_dimension': 1200},  # Стандартные страницы
}

class DocumentParser:
    """
    Универсальный парсер документов.
    Возможности:
    - Сохранение нумерации страниц (через конвертацию в PDF).
    - OCR для изображений и схем внутри документов (Tesseract).
    - Извлечение таблиц из Excel.
    """

    def __init__(self):
        self.mime_to_parser = {
            # PDF
            'application/pdf': self._parse_pdf,
            
            # Офисные документы (через конвертацию в PDF для сохранения страниц)
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._parse_via_pdf_conversion, # docx
            'application/msword': self._parse_via_pdf_conversion, # doc
            'application/rtf': self._parse_via_pdf_conversion, # rtf
            'text/rtf': self._parse_via_pdf_conversion,
            
            # Таблицы
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._parse_excel,
            'application/vnd.ms-excel': self._parse_excel,
            
            # Изображения
            'image/jpeg': self._parse_image,
            'image/png': self._parse_image,
            'image/tiff': self._parse_image,
            'image/bmp': self._parse_image,
            
            'text/plain': self._parse_txt,
            'text/markdown': self._parse_md,
        }
        
        logger.info("✅ DocumentParser initialized (Tesseract only)")

    def parse_file(self, content: bytes, file_name: str, mime_type: str = None, max_pages: int = None) -> List[Dict[str, Any]]:
        """
        Главный метод обработки файла.
        """
        _, ext = os.path.splitext(file_name)
        if not ext:
            ext = mimetypes.guess_extension(mime_type) or ''

        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(file_name)

            # Выбор парсера
            parser = self.mime_to_parser.get(mime_type)
            
            # Фолбэк по расширению
            if not parser:
                if ext.lower() == '.pdf':
                    parser = self._parse_pdf
                elif ext.lower() in ['.docx', '.doc', '.rtf']:
                    parser = self._parse_via_pdf_conversion
                elif ext.lower() in ['.xlsx', '.xls']:
                    parser = self._parse_excel
                elif ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                    parser = self._parse_image
                elif ext.lower() in ['.txt', '.md']:
                    parser = self._parse_txt
            
            if parser:
                logger.info(f"Начало парсинга {file_name} ({parser.__name__})")
                if parser == self._parse_pdf:
                     return parser(temp_file_path, max_pages=max_pages)
                return parser(temp_file_path)
            else:
                logger.warning(f"Неподдерживаемый формат файла: {file_name} ({mime_type})")
                return []

        except Exception as e:
            logger.error(f"Критическая ошибка парсинга {file_name}: {e}")
            return []
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _detect_page_format(self, image: Image.Image) -> Tuple[str, int]:
        """Определяет формат страницы и возвращает максимальный размер"""
        max_size = max(image.size)
        
        if max_size >= ADAPTIVE_SIZE_THRESHOLDS['A2']['max_size']:
            return 'A2', ADAPTIVE_SIZE_THRESHOLDS['A2']['max_dimension']
        elif max_size >= ADAPTIVE_SIZE_THRESHOLDS['A3']['max_size']:
            return 'A3', ADAPTIVE_SIZE_THRESHOLDS['A3']['max_dimension']
        else:
            return 'A4', ADAPTIVE_SIZE_THRESHOLDS['A4']['max_dimension']

    def _optimize_image_adaptive(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Адаптивная оптимизация изображения в зависимости от размера"""
        page_format, max_dim = self._detect_page_format(image)
        
        # Downscaling если нужно
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        
        # Grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        logger.debug(f"Adaptive OCR: {page_format} format, scaled to max {max_dim}px")
        return image, page_format

    def _extract_text_with_tesseract(self, image: Image.Image, lang='rus+eng') -> str:
        """
        Extract text using Tesseract with adaptive scaling.
        """
        try:
            # Применяем адаптивное масштабирование
            image, page_format = self._optimize_image_adaptive(image)
            
            # PSM 3 лучше для таблиц (автоматическое определение структуры)
            custom_config = r'--oem 1 --psm 3'
            text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
            
            logger.debug(f"Tesseract extracted {len(text)} chars from {page_format} page")
            return text.strip()

        except Exception as e:
            logger.warning(f"Ошибка OCR (Tesseract): {e}")
            return ""

    def _is_garbage(self, text: str) -> bool:
        """
        Проверяет, является ли извлеченный текст 'мусором'.
        """
        if not text:
            return False
            
        text_len = len(text)
        if text_len < 20:
            return False # Слишком короткий
            
        # Разрешенные символы:
        # - Кириллица: \u0400-\u04FF (в регексе [а-яА-ЯёЁ])
        # - Латиница (Basic): a-zA-Z
        # - Цифры: 0-9
        # - Пунктуация и пробелы
        allowed_pattern = re.compile(r'[а-яА-ЯёЁa-zA-Z0-9\s.,;:!?()\[\]{}"\'\-_+=/\\|%@#№$€£*<>&]')
        
        # Находим все символы, которые НЕ подходят под паттерн
        garbage_chars_count = len([c for c in text if not allowed_pattern.match(c)])
        
        garbage_ratio = garbage_chars_count / text_len
        
        # Если мусора больше 10%
        if garbage_ratio > 0.1:
            return True
            
        # Дополнительная проверка на кириллицу
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        
        if text_len > 50:
             if cyrillic_count / text_len < 0.05 and latin_count / text_len < 0.5:
                 return True

        return False

    def _is_diagram_signature(self, text: str) -> bool:
        """
        Проверяет, является ли текст только подписью чертежа (штампом).
        Обычно это означает, что основной контент - графический, и PyMuPDF извлек только текст штампа.
        """
        if not text or len(text) > 500: # Если текста много, это вряд ли только штамп
            return False
            
        # Ключевые слова из штампов ГОСТ
        keywords = [
            "Изм.", "Кол.уч.", "Лист", "№ док.", "Подп.", "Дата", 
            "Разраб.", "Пров.", "Т.контр.", "Н.контр.", "Утв.", 
            "Стадия", "Листов", "Масштаб", "Формат"
        ]
        
        found_keywords = 0
        for kw in keywords:
            if kw in text:
                found_keywords += 1
        
        # Если найдено 3+ ключевых слова и текст короткий - это штамп
        if found_keywords >= 3:
            return True
            
        return False

    def _parse_pdf(self, file_path: str, max_pages: int = None) -> List[Dict[str, Any]]:
        """
        Парсинг PDF с таймаутом на страницу (30 секунд).
        Страницы, которые не удалось обработать, логируются в failed_pages.jsonl.
        При 3 подряд таймаутах - остановка обработки документа.
        """
        MAX_TIMEOUT_PER_PAGE = 30  # 30 секунд на страницу
        MAX_CONSECUTIVE_TIMEOUTS = 3  # Максимум 3 подряд таймаута
        
        document_name = os.path.basename(file_path)
        content = []
        consecutive_timeouts = 0  # Счетчик последовательных таймаутов
        failed_pages_list = []  # Список пропущенных страниц
        
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                start_time = time.time()
                
                try:
                    with timeout(MAX_TIMEOUT_PER_PAGE):
                        # 1. Извлекаем обычный текст
                        page_text = page.get_text()
                        
                        # Проверка: пустой текст, мусор или только штамп чертежа
                        is_garbage = self._is_garbage(page_text)
                        is_signature = self._is_diagram_signature(page_text)
                        ocr_used = False  # Флаг для отслеживания использования OCR
                        
                        if not page_text.strip() or is_garbage or is_signature:
                            if not page_text.strip():
                                reason = "пустой текст"
                            elif is_garbage:
                                reason = "мусорный текст"
                            else:
                                reason = "только штамп чертежа"
                            
                            logger.warning(f"Обнаружен {reason} на стр. {page_num + 1}. Применяем полный OCR страницы.")
                            try:
                                # Конвертируем страницу PDF в изображение
                                images = convert_from_path(
                                    file_path, 
                                    first_page=page_num+1, 
                                    last_page=page_num+1, 
                                    dpi=ADAPTIVE_OCR_DPI,  # 250 DPI для качественного распознавания таблиц
                                    fmt='jpeg',
                                    grayscale=True
                                )
                                if images:
                                    # OCR
                                    page_text = self._extract_text_with_tesseract(images[0])
                                    images[0].close()
                                    del images
                                    ocr_used = True
                                    logger.info(f"OCR выполнен успешно для стр. {page_num + 1}")
                            except TimeoutError:
                                raise
                            except Exception as ocr_e:
                                logger.error(f"Ошибка полного OCR страницы {page_num + 1}: {ocr_e}")
                                # Если OCR упал, оставляем оригинальный текст (даже если это штамп)
                                # или пустую строку, если это мусор/пусто
                                if is_garbage or not page_text.strip():
                                    page_text = ""

                        # Логируем извлеченный текст (даже если пустой)
                        logger.info(f"--- Page {page_num + 1} Main Text ({len(page_text)} chars) ---\n{page_text[:500] if len(page_text) > 500 else page_text}\n{'...(truncated)' if len(page_text) > 500 else ''}-----------------------------")
                        
                        # Добавляем основной текст страницы
                        if page_text.strip():
                            content.append({
                                'content': page_text.strip(),
                                'page_number': page_num + 1,
                                'type': 'ocr_text' if ocr_used else 'text'
                            })
                        
                        # Сбрасываем счетчик таймаутов при успешной обработке
                        consecutive_timeouts = 0
                        
                        processing_time = time.time() - start_time
                        logger.info(f"Страница {page_num + 1} обработана за {processing_time:.2f}с")
                        
                except TimeoutError:
                    consecutive_timeouts += 1
                    processing_time = time.time() - start_time
                    error_msg = f"Timeout ({MAX_TIMEOUT_PER_PAGE}s exceeded)"
                    logger.error(f"⏱️ Таймаут на странице {page_num + 1}: {processing_time:.2f}с (подряд: {consecutive_timeouts})")
                    
                    failed_pages_list.append(page_num + 1)
                    
                    # Логируем в failed_pages.jsonl
                    failed_pages_logger.log_failed_page(
                        document_name=document_name,
                        page_number=page_num + 1,
                        reason=error_msg,
                        processing_time=processing_time
                    )
                    
                    # Добавляем заметку о пропущенной странице
                    content.append({
                        'content': f"[СТРАНИЦА {page_num + 1} НЕ ОБРАБОТАНА: Превышен таймаут {MAX_TIMEOUT_PER_PAGE}с]",
                        'page_number': page_num + 1,
                        'type': 'failed'
                    })
                    
                    # Проверяем, если 3 подряд таймаута - останавливаем обработку
                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                        logger.error(f"⚠️ Остановка обработки {document_name}: 3 подряд таймаута")
                        
                        # Логируем все оставшиеся страницы как пропущенные
                        failed_pages_logger.log_failed_page(
                            document_name=document_name,
                            page_number="все страницы",
                            reason=f"3 consecutive timeouts, stopped at page {page_num + 1}",
                            processing_time=0
                        )
                        break  # Останавливаем обработку документа
                    
                except Exception as page_error:
                    consecutive_timeouts = 0  # Сбрасываем при обычных ошибках
                    processing_time = time.time() - start_time
                    error_msg = f"Error: {str(page_error)[:200]}"
                    logger.error(f"❌ Ошибка на странице {page_num + 1}: {error_msg}")
                    
                    # Логируем в failed_pages.jsonl
                    failed_pages_logger.log_failed_page(
                        document_name=document_name,
                        page_number=page_num + 1,
                        reason=error_msg,
                        processing_time=processing_time
                    )
                    
                    # Продолжаем обработку следующих страниц
                    content.append({
                        'content': f"[СТРАНИЦА {page_num + 1} НЕ ОБРАБОТАНА: {error_msg}]",
                        'page_number': page_num + 1,
                        'type': 'failed'
                    })
            
            # Группируем контент по страницам
            page_contents = {}
            for item in content:
                pn = item['page_number']
                if pn not in page_contents:
                    page_contents[pn] = []
                page_contents[pn].append(item['content'])
                
            # Формируем итоговый список
            result = []
            for pn in sorted(page_contents.keys()):
                full_text = "\n\n".join(page_contents[pn])
                result.append({
                    'content': full_text,
                    'page_number': pn,
                    'type': 'mixed'
                })
            
            return result

        except Exception as e:
            logger.error(f"Ошибка PyMuPDF: {e}")
            return []


    def _parse_via_pdf_conversion(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Конвертирует DOCX/DOC/RTF в PDF, затем парсит PDF.
        """
        pdf_path = None
        try:
            pdf_fd, pdf_path = tempfile.mkstemp(suffix='.pdf')
            os.close(pdf_fd)

            logger.info("Конвертация файла в PDF...")
            pypandoc.convert_file(
                file_path, 
                'pdf', 
                outputfile=pdf_path, 
                extra_args=['--pdf-engine=xelatex'] 
            )

            return self._parse_pdf(pdf_path)

        except Exception as e:
            logger.error(f"Ошибка конвертации в PDF: {e}")
            return []
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

    def _parse_image(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг отдельного файла изображения."""
        try:
            image = Image.open(file_path)
            text = self._extract_text_with_tesseract(image)
            return [{
                'content': text,
                'page_number': 1,
                'type': 'image'
            }]
        except Exception as e:
            logger.error(f"Ошибка парсинга изображения: {e}")
            return []

    def _parse_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг Excel в Markdown-таблицы."""
        content = []
        try:
            xls = pd.ExcelFile(file_path)
            for i, sheet_name in enumerate(xls.sheet_names):
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                if not df.empty:
                    text = df.to_markdown(index=False)
                    content.append({
                        'content': f"Таблица (Лист: {sheet_name}):\n{text}",
                        'page_number': i + 1,
                        'sheet_name': sheet_name,
                        'type': 'table'
                    })
            return content
        except Exception as e:
            logger.error(f"Ошибка Excel: {e}")
            return []

    def _parse_md(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            html = markdown.markdown(md_text)
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")
            return [{'content': text.strip(), 'page_number': 1, 'type': 'text'}]
        except Exception as e:
            logger.error(f"Ошибка MD: {e}")
            return []

    def _parse_txt(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return [{'content': text.strip(), 'page_number': 1, 'type': 'text'}]
        except Exception as e:
            logger.error(f"Ошибка TXT: {e}")
            return []

document_parser = DocumentParser()