import re
import markdown
from bs4 import BeautifulSoup
import pypandoc
import nltk
from docx2python import docx2python
from io import BytesIO
import fitz  # PyMuPDF
import pandas as pd
import pytesseract
import tiktoken
from PIL import Image
from src.app.logging_config import get_logger
import tempfile
import os
logger = get_logger(__name__)

def parse_pdf(content: bytes) -> list[tuple[str, int]]:
    """Parses a .pdf file and returns its text content, including OCR for images."""
    pdf_document = fitz.open(stream=content, filetype="pdf")
    text_by_page = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Extract text
        page_text = page.get_text("text")

        # Check for images and extract text using OCR
        image_list = page.get_images(full=True)
        if image_list:
            logger.info(f"Found {len(image_list)} images on page {page_num + 1}")
                            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                                    
                    # Check if image is valid for OCR
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Save image to temporary file for OCR processing
                        temp_dir = tempfile.gettempdir()
                        temp_img_path = os.path.join(temp_dir, f"pdf_image_{page_num}_{img_index}.png")
                        pix.save(temp_img_path)
                        
                        # Extract text using OCR
                        ocr_text = pytesseract.image_to_string(temp_img_path, lang='rus')
                        if ocr_text and len(ocr_text.strip()) > 10:  # Only add if meaningful text
                            page_text += "\n" + ocr_text
                            logger.info(f"OCR extracted {len(ocr_text)} characters from image on page {page_num + 1}")
                        else:
                            logger.debug(f"OCR returned insufficient text from image on page {page_num + 1}")
                        
                        # Clean up temporary file
                        try:
                            os.remove(temp_img_path)
                        except:
                            pass
                    
                    pix = None

                except Exception as e:
                    logger.info(f"Error processing image on page {page_num + 1}: {e}")
                    pix = None
                finally:
                        # Гарантирует, что файл не останется, даже если была ошибка.
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)
        
            text_by_page.append((page_text, page_num + 1))

    return text_by_page

def parse_docx(content) -> list[tuple[str, int]]:
    """Parses a .docx file and returns its text content."""
    result = docx2python(BytesIO(content))
    
    # Text grouped by page
    text_by_page = []
    
    # Iterate over pages
    for page in result.body:
        page_text = ""
        # Iterate over paragraphs on page
        for paragraph in page:
            page_text += paragraph.text + "\n"
        # Add page text and page number
        text_by_page.append((page_text, page.page_num))
        
    return text_by_page

def parse_doc(content) -> list[tuple[str, int | None]]:
    """Parses a .doc file and returns its text content."""
    return [(pypandoc.convert_text(content, 'plain', format='doc'), None)]

def parse_rtf(content) -> list[tuple[str, int | None]]:
    """Parses an .rtf file and returns its text content."""
    return [(pypandoc.convert_text(content, 'plain', format='rtf'), None)]

def parse_md(content: str) -> list[tuple[str, int | None]]:
 
    # 1. Convert Markdown to HTML
    html = markdown.markdown(content)
    
    # 2. Parse the HTML
    soup = BeautifulSoup(html, "html.parser")
    
    text_content = soup.get_text(separator="\n", strip=True)
    
    return [(text_content, None)]

def parse_excel(content: bytes) -> list[tuple[str, int | None]]:
    """Parses an Excel file (.xls, .xlsx) and returns its text content."""
    xls = pd.ExcelFile(BytesIO(content))
    full_text = ""
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        full_text += df.to_string()
    return [(full_text, None)]


def parse_image(content: bytes) -> list[tuple[str, int | None]]:
    """Parses an image file and returns its text content using OCR."""
    try:
        image = Image.open(BytesIO(content))
        ocr_text = pytesseract.image_to_string(image, lang="rus")
        return [(ocr_text, None)]
    except Exception as e:
        print(f"Error processing image: {e}")
        return []


def parse_txt(content: str) -> list[tuple[str, int | None]]:
    """Parses a .txt file and returns its text content."""
    return [(content, None)]


def clean_text(text: str) -> str:
    """Cleans the text by removing HTML/Markdown artifacts and normalizing whitespace."""
    cleaned_text = re.sub(r'<[^>]+>', ' ', text)
    cleaned_text = re.sub(r'&[a-z]+;|&#x?[0-9a-f]+;', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text).strip()
    return cleaned_text

def split_text_into_chunks(
    text: str, chunk_size: int = 200, overlap: int = 50
) -> list[str]:
    """
    Splits the text into chunks of a specified token size with overlap, using a hierarchical approach.
    The text is first split by paragraphs. If a paragraph is larger than the chunk size,
    it is further split by sentences.
    """
    if not text:
        return []
  
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split text into paragraphs
    pattern = r'\n\s*\n+'
    paragraphs = re.split(pattern, text)
    logger.info('paragraphs[:7]', paragraphs)
    # Удаляем пустые элементы, которые могут появиться из-за пробелов в начале/конце файла
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    for paragraph in paragraphs:
        # Encode paragraph to tokens
        paragraph_tokens = tokenizer.encode(paragraph)
        
        # If paragraph is within chunk size, treat it as a whole chunk
        if len(paragraph_tokens) <= chunk_size:
            if paragraph.strip():
                chunks.append(paragraph)
        else:
            # If paragraph is too long, split it by sentences
            sentences = nltk.sent_tokenize(paragraph, language="russian")
            current_chunk_tokens = []
            
            for sentence in sentences:
                sentence_tokens = tokenizer.encode(sentence)
                
                # If adding the new sentence exceeds the chunk size, process the current chunk
                if len(current_chunk_tokens) + len(sentence_tokens) > chunk_size:
                    if current_chunk_tokens:
                        # Decode tokens to string and add to chunks
                        chunk_text = tokenizer.decode(current_chunk_tokens).strip()
                        if chunk_text:
                            chunks.append(chunk_text)
                        
                        # Start a new chunk with an overlap
                        current_chunk_tokens = current_chunk_tokens[-overlap:]
                    else:
                        # Handle cases where a single sentence is longer than the chunk size
                        current_chunk_tokens = sentence_tokens
                
                current_chunk_tokens.extend(sentence_tokens)

            # Add the last remaining chunk
            if current_chunk_tokens:
                chunk_text = tokenizer.decode(current_chunk_tokens).strip()
                if chunk_text:
                    chunks.append(chunk_text)

    # Apply overlap to the final list of chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            # Get the tokens for the previous and current chunks
            prev_chunk_tokens = tokenizer.encode(chunks[i - 1])
            current_chunk_tokens = tokenizer.encode(chunks[i])
            
            # Create overlap
            overlap_tokens = prev_chunk_tokens[-overlap:]
            
            # Combine overlap with the current chunk
            overlapped_chunk_tokens = overlap_tokens + current_chunk_tokens
            
            # Decode back to string
            overlapped_chunks.append(tokenizer.decode(overlapped_chunk_tokens))
            
        return overlapped_chunks

    return chunks
