import os
import logging
import tempfile
import mimetypes
import io
import re
import time
from typing import List, Dict, Any

# Библиотеки для обработки форматов
import pypandoc
import fitz  # PyMuPDF
import pandas as pd
import markdown
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path

# Наши утилиты
from .page_processing_utils import timeout, TimeoutError
from .failed_pages_logger import failed_pages_logger

# Настройка логирования
from src.app.logging_config import get_logger
logger = get_logger(__name__)

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

    def _extract_text_with_tesseract(self, image: Image.Image, lang='rus+eng') -> str:
        """
        Extract text using Tesseract with confidence filtering.
        Only returns text if confidence > 70 (0.7).
        """
        try:
            # Используем image_to_data для получения уверенности
            # config='--psm 6' предполагаем единый блок текста
            custom_config = r'--oem 1 --psm 6'
            
            data = pytesseract.image_to_data(image, lang=lang, config=custom_config, output_type=Output.DICT)
            
            extracted_words = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                # data['conf'][i] может быть -1 для блоков без текста
                conf = float(data['conf'][i])
                text = data['text'][i].strip()
                
                # Фильтруем по уверенности > 70 (это 0.7 в Tesseract scale 0-100)
                if int(conf) > 70 and text:
                    extracted_words.append(text)
            
            # Если слов мало или ничего не нашли
            if not extracted_words:
                return ""
                
            # Собираем текст обратно (просто через пробел, так как структура теряется при фильтрации слов)
            # Для более сложной структуры нужно анализировать block_num, par_num, line_num
            # Но для "вытащить текст из картинки" этого часто достаточно.
            # Попробуем сохранить переносы строк, если они были в исходных блоках?
            # image_to_data дает line_num.
            
            # Пересборка с учетом строк:
            lines = {}
            for i in range(n_boxes):
                conf = float(data['conf'][i])
                text = data['text'][i].strip()
                if int(conf) > 70 and text:
                    line_num = data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = []
                    lines[line_num].append(text)
            
            sorted_lines = sorted(lines.keys())
            final_text = "\n".join([" ".join(lines[ln]) for ln in sorted_lines])
            
            logger.debug(f"Tesseract extracted {len(final_text)} chars with conf > 70")
            return final_text.strip()

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

    def _parse_pdf(self, file_path: str, max_pages: int = None) -> List[Dict[str, Any]]:
        """
        Парсинг PDF с таймаутом на страницу (120 секунд).
        Страницы, которые не удалось обработать, логируются в failed_pages.jsonl
        """
        MAX_TIMEOUT_PER_PAGE = 120  # 2 минуты на страницу
        
        document_name = os.path.basename(file_path)
        content = []
        
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
                        
                        # Проверка на мусор
                        if self._is_garbage(page_text):
                            logger.warning(f"Обнаружен мусорный текст на стр. {page_num + 1}. Применяем полный OCR страницы.")
                            try:
                                # Конвертируем страницу PDF в изображение
                                images = convert_from_path(
                                    file_path, 
                                    first_page=page_num+1, 
                                    last_page=page_num+1, 
                                    dpi=70,
                                    fmt='jpeg',
                                    grayscale=True
                                )
                                if images:
                                    # OCR
                                    page_text = self._extract_text_with_tesseract(images[0])
                                    images[0].close()
                                    del images
                                    logger.info(f"OCR выполнен успешно для стр. {page_num + 1}")
                            except TimeoutError:
                                raise
                            except Exception as ocr_e:
                                logger.error(f"Ошибка полного OCR страницы {page_num + 1}: {ocr_e}")
                                page_text = "" 

                        # 2. Ищем изображения на странице (схемы, сканы) - ТОЛЬКО если мы не делали полный OCR
                        if not self._is_garbage(page.get_text()):
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
                                        extracted_text = self._extract_text_with_tesseract(image)
                                        if len(extracted_text) > 10: 
                                            content.append({
                                                'content': f"[Текст из изображения/схемы]:\n{extracted_text}",
                                                'page_number': page_num + 1,
                                                'type': 'image_text'
                                            })
                                    except TimeoutError:
                                        raise
                                    except Exception as img_e:
                                        logger.warning(f"Ошибка обработки изображения на стр. {page_num + 1}: {img_e}")

                        # Добавляем основной текст страницы
                        if page_text.strip():
                            content.append({
                                'content': page_text.strip(),
                                'page_number': page_num + 1,
                                'type': 'text' if not self._is_garbage(page.get_text()) else 'ocr_text'
                            })
                        
                        processing_time = time.time() - start_time
                        logger.info(f"Страница {page_num + 1} обработана за {processing_time:.2f}с")
                        
                except TimeoutError:
                    processing_time = time.time() - start_time
                    error_msg = f"Timeout ({MAX_TIMEOUT_PER_PAGE}s exceeded)"
                    logger.error(f"⏱️ Таймаут на странице {page_num + 1}: {processing_time:.2f}с")
                    
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
                    
                except Exception as page_error:
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

    def _parse_pdf_old(self, file_path: str, max_pages: int = None) -> List[Dict[str, Any]]:
        """
        Парсинг PDF: текст + изображения (OCR).
        СТАРАЯ ВЕРСИЯ БЕЗ ТАЙМАУТА (оставлена как резерв)
        """
        content = []
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                if max_pages and page_num >= max_pages:
                    logger.info(f"Достигнут лимит страниц ({max_pages}). Остановка парсинга.")
                    break
                
                # 1. Извлекаем обычный текст
                page_text = page.get_text()
                
                # Проверка на мусор
                if self._is_garbage(page_text):
                    logger.warning(f"Обнаружен мусорный текст на стр. {page_num + 1}. Применяем полный OCR страницы.")
                    try:
                        # Конвертируем страницу PDF в изображение
                        # DPI=70 для скорости, fmt='jpeg' (хотя convert_from_path возвращает PIL, внутренне это может влиять на скорость рендеринга poppler)
                        images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1, dpi=70)
                        if images:
                            # OCR
                            page_text = self._extract_text_with_tesseract(images[0])
                            images[0].close()
                            del images
                            logger.info(f"OCR выполнен успешно для стр. {page_num + 1}")
                    except Exception as ocr_e:
                        logger.error(f"Ошибка полного OCR страницы {page_num + 1}: {ocr_e}")
                        page_text = "" 

                # 2. Ищем изображения на странице (схемы, сканы) - ТОЛЬКО если мы не делали полный OCR
                # Если текст был мусором, мы его заменили через OCR всей страницы.
                # Проверяем снова, является ли текущий page_text (возможно обновленный) мусором? 
                # Нет, если мы сделали OCR, то у нас есть текст.
                # Но если мы НЕ делали OCR (текст был норм), то ищем картинки.
                
                # Логика: если мы НЕ вызывали convert_from_path, то ищем картинки.
                # Но у нас нет флага. Проще проверить: если page.get_text() был норм, то ищем картинки.
                
                if not self._is_garbage(page.get_text()):
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
                                extracted_text = self._extract_text_with_tesseract(image)
                                if len(extracted_text) > 10: 
                                    content.append({
                                        'content': f"[Текст из изображения/схемы]:\n{extracted_text}",
                                        'page_number': page_num + 1,
                                        'type': 'image_text'
                                    })
                            except Exception as img_e:
                                logger.warning(f"Ошибка обработки изображения на стр. {page_num + 1}: {img_e}")

                # Добавляем основной текст страницы
                if page_text.strip():
                    content.append({
                        'content': page_text.strip(),
                        'page_number': page_num + 1,
                        'type': 'text' if not self._is_garbage(page.get_text()) else 'ocr_text'
                    })
            
            # Объединяем контент страницы? 
            # В оригинале было: content.append({'content': full_page_content ...})
            # Здесь я добавлял image_text отдельно в content list.
            # Лучше следовать структуре оригинала: один элемент на страницу или список элементов?
            # Оригинал возвращал список словарей.
            # Моя новая логика добавляет image_text как отдельные элементы. Это нормально для RAG.
            # Но чтобы сохранить совместимость, лучше склеить, если ожидается один блок на страницу?
            # В оригинале: full_page_content = page_text + ocr_text_list.
            # Давайте вернемся к склеиванию, чтобы не ломать логику чанков (если она завязана на страницы).
            
            # Переделываем сборку контента
            final_content = []
            
            # Группируем по страницам
            page_contents = {} # page_num -> list of text
            
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