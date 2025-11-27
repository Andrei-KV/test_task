import os
import logging
import tempfile
import mimetypes
import io
import re
from typing import List, Dict, Any

# Библиотеки для обработки форматов
import pypandoc
import fitz  # PyMuPDF
import pandas as pd
import markdown
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Настройка логирования
from src.app.logging_config import get_logger
logger = get_logger(__name__)

class DocumentParser:
    """
    Универсальный парсер документов.
    Возможности:
    - Сохранение нумерации страниц (через конвертацию в PDF).
    - OCR для изображений и схем внутри документов.
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
            
            # Текст
            'text/plain': self._parse_txt,
            'text/markdown': self._parse_md,
        }

    def parse_file(self, content: bytes, file_name: str, mime_type: str = None) -> List[Dict[str, Any]]:
        """
        Главный метод обработки файла.
        """
        _, ext = os.path.splitext(file_name)
        if not ext:
            ext = mimetypes.guess_extension(mime_type) or ''

        # Сохраняем во временный файл
        # Это необходимо, так как многие библиотеки (PyMuPDF, python-docx) работают с путями к файлам, а не с байтами в памяти.
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
                ext_lower = ext.lower()
                if ext_lower in ['.docx', '.doc', '.rtf']:
                    parser = self._parse_via_pdf_conversion
                elif ext_lower == '.pdf':
                    parser = self._parse_pdf
                elif ext_lower in ['.xlsx', '.xls']:
                    parser = self._parse_excel
                elif ext_lower in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    parser = self._parse_image
                elif ext_lower in ['.txt', '.md']:
                    parser = self._parse_txt

            if not parser:
                logger.warning(f"Не найден парсер для {file_name} ({mime_type})")
                return []

            logger.info(f"Начало парсинга {file_name} ({parser.__name__})")
            return parser(temp_file_path)

        except Exception as e:
            logger.error(f"Критическая ошибка парсинга {file_name}: {e}")
            return []
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _perform_ocr(self, image: Image.Image, lang='rus+eng') -> str:
        """Вспомогательный метод для OCR."""
        try:
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            logger.warning(f"Ошибка OCR: {e}")
            return ""


    def _is_garbage(self, text: str) -> bool:
        """
        Проверяет, является ли извлеченный текст 'мусором'.
        Улучшенная эвристика:
        1. Считаем отношение "разрешенных" символов (кириллица, латиница, цифры, пунктуация) к общей длине.
        2. Если "мусорных" символов (Extended Latin и прочее) > 10%, считаем текст мусором.
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
        # - Пунктуация и пробелы: \s.,;:!?()[]{}"'-_+=/\|%@#№$€£*<>&
        
        # Паттерн разрешенных символов
        allowed_pattern = re.compile(r'[а-яА-ЯёЁa-zA-Z0-9\s.,;:!?()\[\]{}"\'\-_+=/\\|%@#№$€£*<>&]')
        
        # Находим все символы, которые НЕ подходят под паттерн
        garbage_chars_count = len([c for c in text if not allowed_pattern.match(c)])
        
        garbage_ratio = garbage_chars_count / text_len
        
        # Если мусора больше 10%, то это, вероятно, битая кодировка (Extended Latin и т.д.)
        if garbage_ratio > 0.1:
            return True
            
        # Дополнительная проверка на кириллицу для русскоязычных документов
        # Если текст длинный (>50), а кириллицы < 5% и это не чистая латиница (английский), то подозрительно.
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        
        if text_len > 50:
             # Если мало кириллицы И мало латиницы (например, одни цифры и спецсимволы - это ОК, но если там кракозябры...)
             # Но кракозябры обычно попадают в garbage_ratio.
             # Эта проверка скорее для случаев, когда кракозябры состоят из Basic Latin, но бессмысленны.
             # Пока оставим проверку на отсутствие кириллицы, если это не английский текст.
             if cyrillic_count / text_len < 0.05 and latin_count / text_len < 0.5:
                 # Мало кириллицы и не похоже на английский текст -> вероятно мусор
                 return True

        return False

    def _parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Парсинг PDF: текст + изображения (OCR).
        Если текст страницы похож на мусор, страница конвертируется в изображение и OCR-ится целиком.
        """
        content = []
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                # 1. Извлекаем обычный текст
                page_text = page.get_text()
                
                # Проверка на мусор
                if self._is_garbage(page_text):
                    logger.warning(f"Обнаружен мусорный текст на стр. {page_num + 1}. Применяем полный OCR страницы.")
                    try:
                        # Конвертируем страницу PDF в изображение
                        # Используем pdf2image с пониженным DPI для экономии памяти (200 достаточно для текста)
                        images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1, dpi=200)
                        if images:
                            # Освобождаем память, закрывая doc на время (хотя fitz держит файл открытым)
                            # Лучше просто обработать и удалить ссылку на image сразу
                            page_text = self._perform_ocr(images[0])
                            images[0].close() # Явное закрытие
                            del images
                            logger.info(f"OCR выполнен успешно для стр. {page_num + 1}")
                    except Exception as ocr_e:
                        logger.error(f"Ошибка полного OCR страницы {page_num + 1}: {ocr_e}")
                        page_text = "" 

                # 2. Ищем изображения на странице (схемы, сканы) - ТОЛЬКО если мы не делали полный OCR
                # Если мы сделали полный OCR, то картинки уже распознаны в составе страницы
                ocr_text_list = []
                # Проверяем, делали ли мы полный OCR (если page_text был заменен)
                # Но page_text мог быть заменен на результат OCR, который тоже содержит текст.
                # Флаг garbage_detected был бы удобнее, но мы уже внутри цикла.
                # Просто проверим, не вызывали ли мы convert_from_path
                
                # Упрощение: если текст был мусором, мы его заменили. 
                # Если мы заменили текст через OCR всей страницы, то отдельные картинки извлекать не нужно.
                
                # Но fitz.open открыт.
                # Давайте сделаем так: если текст нормальный, то ищем картинки.
                # Если текст был мусором, мы уже получили всё через OCR.
                
                if not self._is_garbage(page.get_text()): # Проверяем ИСХОДНЫЙ текст страницы снова (немного неэффективно, но надежно)
                     image_list = page.get_images(full=True)
                     if image_list:
                        logger.info(f"Найдено {len(image_list)} изображений на стр. {page_num + 1}")
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                
                                # Загружаем в PIL
                                image = Image.open(io.BytesIO(image_bytes))
                                
                                # Фильтр: пропускаем слишком маленькие иконки/линии
                                if image.width < 50 or image.height < 50:
                                    continue

                                # Выполняем OCR
                                extracted_text = self._perform_ocr(image)
                                if len(extracted_text) > 10: # Игнорируем мусор
                                    ocr_text_list.append(f"[Текст из изображения/схемы]:\n{extracted_text}")
                            except Exception as img_e:
                                logger.warning(f"Ошибка обработки изображения на стр. {page_num + 1}: {img_e}")

                # Объединяем текст страницы и текст из картинок
                full_page_content = page_text
                if ocr_text_list:
                    full_page_content += "\n\n" + "\n---\n".join(ocr_text_list)

                if full_page_content.strip():
                    content.append({
                        'content': full_page_content.strip(),
                        'page_number': page_num + 1,
                        'type': 'mixed'
                    })
            return content
        except Exception as e:
            logger.error(f"Ошибка PyMuPDF: {e}")
            return []

    def _parse_via_pdf_conversion(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Конвертирует DOCX/DOC/RTF в PDF, затем парсит PDF.
        Это позволяет извлечь и текст, и картинки (через OCR в _parse_pdf), и сохранить номера страниц.
        """
        pdf_path = None
        try:
            pdf_fd, pdf_path = tempfile.mkstemp(suffix='.pdf')
            os.close(pdf_fd)

            logger.info("Конвертация файла в PDF...")
            # Используем xelatex или wkhtmltopdf движок
            pypandoc.convert_file(
                file_path, 
                'pdf', 
                outputfile=pdf_path, 
                extra_args=['--pdf-engine=xelatex'] 
            )

            # Рекурсивно вызываем парсер PDF
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
            text = self._perform_ocr(image)
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
                # Чистка пустых данных
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                if not df.empty:
                    # Конвертация в Markdown сохраняет структуру таблицы
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