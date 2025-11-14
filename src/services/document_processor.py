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
    cnt = 0
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Extract text
        page_text = page.get_text("text")


        # # Check for images and extract text using OCR
        # image_list = page.get_images(full=True)
        # if image_list:
        #     logger.info(f"Found {len(image_list)} images on page {page_num + 1}")
                            
        #     for img_index, img in enumerate(image_list):
        #         try:
        #             xref = img[0]
        #             pix = fitz.Pixmap(pdf_document, xref)
                                    
        #             # Check if image is valid for OCR
        #             if pix.n - pix.alpha < 4:  # GRAY or RGB
        #                 # Save image to temporary file for OCR processing
        #                 temp_dir = tempfile.gettempdir()
        #                 temp_img_path = os.path.join(temp_dir, f"pdf_image_{page_num}_{img_index}.png")
        #                 pix.save(temp_img_path)
                        
        #                 # Extract text using OCR
        #                 ocr_text = pytesseract.image_to_string(temp_img_path, lang='rus')
        #                 if ocr_text and len(ocr_text.strip()) > 10:  # Only add if meaningful text
        #                     page_text += "\n" + ocr_text
        #                     logger.info(f"OCR extracted {len(ocr_text)} characters from image on page {page_num + 1}")
        #                 else:
        #                     logger.debug(f"OCR returned insufficient text from image on page {page_num + 1}")
                        
        #                 # Clean up temporary file
        #                 try:
        #                     os.remove(temp_img_path)
        #                 except:
        #                     pass
                    
        #             pix = None

        #         except Exception as e:
        #             logger.info(f"Error processing image on page {page_num + 1}: {e}")
        #             pix = None
        #         finally:
        #                 # Гарантирует, что файл не останется, даже если была ошибка.
        #                 if os.path.exists(temp_img_path):
        #                     os.remove(temp_img_path)
        
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

    return cleaned_text

def split_text_into_chunks(
    text: str, chunk_size: int = 200, overlap: int = 50, 
    previous_overlap_sentences: list[str] = None
) -> tuple[list[str], list[str]]:
    """
    Splits the text into chunks of a specified token size with overlap.
    The text is first split into paragraphs. Paragraphs are cleaned and filtered.
    Chunks are then created by greedily adding paragraphs until the chunk size is reached.
    Overlap between chunks is also paragraph-based.
    """
    if not text:
        return [], previous_overlap_sentences

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 1. Split, clean, and filter paragraphs
    pattern = r'\n\s*\n+'
    paragraphs = re.split(pattern, text)
    cleaned_paragraphs = []
    for p in paragraphs:
        cleaned_p = clean_text(p)
        if len(cleaned_p.split()) >= 3:
            cleaned_paragraphs.append(cleaned_p)

    if not cleaned_paragraphs:
        return [], previous_overlap_sentences
    
    
    docs_for_chunking = []
    for p in cleaned_paragraphs:
        sentences = nltk.sent_tokenize(p, language="russian") 

        # Adjust sentences to merge short numbered items with the next sentence
        adjusted_sentences = []
        i = 0
        while i < len(sentences):
            current_sentence = sentences[i]
            
            # Check if the sentence is short and resembles a numbered item (e.g., "15.")
            is_number_only = re.match(r'^\s*\d+\.\s*$', current_sentence)
            
            # Also check for overly short sentences (e.g., one or two words)
            is_too_short = len(current_sentence.split()) <= 2
            
            if (is_number_only or is_too_short) and (i + 1 < len(sentences)):
                # If it's a number item or too short, merge it with the next sentence.
                sentences[i + 1] = current_sentence.strip() + ' ' + sentences[i + 1].strip()
            else:
                # Otherwise, add the sentence as is
                adjusted_sentences.append(current_sentence.strip())
            
            i += 1
            
        sentences = adjusted_sentences
        docs_for_chunking.extend(sentences)
    
    if previous_overlap_sentences is None:
        previous_overlap_sentences = []

    if not docs_for_chunking:
        return [], previous_overlap_sentences
    
    if previous_overlap_sentences is None:
        previous_overlap_sentences = []

    docs_for_chunking = previous_overlap_sentences + docs_for_chunking

    # 3. Create chunks with greedy approach, using sentences as base units
    chunks = []
    # current_sentences_in_chunk: list of sentences that make up the current chunk
    current_sentences_in_chunk = [] 
    current_chunk_text = ""
    
    for sentence in docs_for_chunking:
        
        # 1. Check if the next sentence fits
        next_chunk_text = current_chunk_text + ' ' + sentence if current_chunk_text else sentence
        
        if len(tokenizer.encode(next_chunk_text)) > chunk_size:
            
            # The chunk is full, finalize it
            if current_chunk_text:
                chunks.append(current_chunk_text.strip())

            # 2. Form the overlap for the new chunk (50 tokens) from previous sentences
            overlap_sentences = []
            overlap_tokens = 0
            
            # Search backwards, taking sentences until at least 50 tokens are accumulated
            for s in reversed(current_sentences_in_chunk):
                s_tokens_count = len(tokenizer.encode(s))
                
                # Condition: Adding this sentence to the overlap + the current sentence 
                # must not exceed the chunk_size (200)
                if overlap_tokens + s_tokens_count + len(tokenizer.encode(sentence)) <= chunk_size:
                    
                    overlap_sentences.insert(0, s) # Insert at the start to maintain order
                    overlap_tokens += s_tokens_count
                    
                    # If the minimum overlap (50 tokens) is reached, stop to minimize duplication.
                    if overlap_tokens >= overlap:
                        break
                
            # 3. Start the new chunk with the overlap and add the current sentence
            
            # New sentences list: Overlap sentences + current sentence
            new_chunk_sentences = overlap_sentences + [sentence]
            
            # Обновляем текущий текст чанка
            current_chunk_text = " ".join(new_chunk_sentences).strip()
            
            # Обновляем список предложений, составляющих новый чанк
            current_sentences_in_chunk = new_chunk_sentences

        else:
            # 4. Add the sentence to the current chunk
            current_chunk_text = next_chunk_text.strip()
            current_sentences_in_chunk.append(sentence)

    # 5. Finalize the last chunk
    if current_chunk_text and (not chunks or current_chunk_text.strip() != chunks[-1].strip()):
        chunks.append(current_chunk_text.strip())

    # 6. Determine the tail (overlap for the next page)
    tail_sentences = []
    tail_tokens = 0
    
    for s in reversed(current_sentences_in_chunk):
        s_tokens_count = len(tokenizer.encode(s))
        if tail_tokens + s_tokens_count <= chunk_size: 
            tail_sentences.insert(0, s) 
            tail_tokens += s_tokens_count
        
        if tail_tokens >= overlap:
            break
            
    logger.info(f'FLAGchunks {chunks}')
    return chunks, tail_sentences

 
