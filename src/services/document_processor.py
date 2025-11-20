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

def parse_docx(content: bytes) -> list[tuple[str, int]]:
    """
    Parses a .docx file by first converting it to PDF to ensure accurate page numbering,
    then processes the resulting PDF.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf_path = temp_pdf.name
        
        pypandoc.convert_text(content, 'pdf', format='doc', outputfile=temp_pdf_path, extra_args=['--pdf-engine=xelatex'])
        
        with open(temp_pdf_path, "rb") as f:
            pdf_content = f.read()
            
        os.remove(temp_pdf_path)
        
        return parse_pdf(pdf_content)
    except OSError as e:
        logger.error(f"Error converting .docx to PDF. Pandoc and a LaTeX engine (e.g., TeX Live) must be installed. Error: {e}")
        logger.warning("Falling back to legacy docx2python parser for .docx file. Page numbers may be inaccurate.")
        # Fallback to the old method if pandoc fails
        try:
            result = docx2python(BytesIO(content))
            text_by_page = []
            for page in result.body:
                page_text = ""
                for paragraph in page:
                    page_text += paragraph.text + "\n"
                text_by_page.append((page_text, page.page_num))
            return text_by_page
        except Exception as fallback_e:
            logger.error(f"Fallback .docx parsing with docx2python also failed: {fallback_e}")
            return []


def parse_doc(content: bytes) -> list[tuple[str, int]]:
    """
    Parses a .doc file by first trying to convert it as DOCX, then as RTF,
    to handle different .doc format variations.
    """
    try:
        # Try parsing as DOCX first
        return parse_doc_or_rtf(content, 'docx')
    except Exception as e:
        logger.warning(f"Failed to parse .doc as DOCX: {e}. Trying as RTF.")
        try:
            # Fallback to parsing as RTF
            return parse_doc_or_rtf(content, 'rtf')
        except Exception as e2:
            logger.error(f"Failed to parse .doc as both DOCX and RTF: {e2}")
            return []

def parse_doc_or_rtf(content: bytes, file_format: str) -> list[tuple[str, int]]:
    """Helper function to convert DOCX or RTF to PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf_path = temp_pdf.name
    
    pypandoc.convert_text(content, 'pdf', format=file_format, outputfile=temp_pdf_path, extra_args=['--pdf-engine=xelatex'])
    
    with open(temp_pdf_path, "rb") as f:
        pdf_content = f.read()
        
    os.remove(temp_pdf_path)
    
    return parse_pdf(pdf_content)


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
    text: str, chunk_size: int = 500, overlap: int = 100, 
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
        logger.debug(f'FLAGparagraph before cleaning [:20]: {p[:20]}')
        cleaned_p = clean_text(p)
        if len(cleaned_p.split()) >= 3:
            cleaned_paragraphs.append(cleaned_p)

    if not cleaned_paragraphs:
        return [], previous_overlap_sentences
    
    
    docs_for_chunking = []
    for p in cleaned_paragraphs:
       
        sentences = nltk.sent_tokenize(p, language="russian") 
        # logger.debug(f'FLAGsentences before adjustment [:]: {sentences}')
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
    

    docs_for_chunking = previous_overlap_sentences + docs_for_chunking
    # logger.info(f'FLAGdocs_for_chunking: {docs_for_chunking}')
    # 3. Create chunks with greedy approach, using sentences as base units
    chunks = []
    # current_sentences_in_chunk: list of sentences that make up the current chunk
    current_sentences_in_chunk = [] 
    current_chunk_text = ""
    
    for sentence in docs_for_chunking:
        current_sentences_in_chunk.append(sentence)
        # 1. Check if the next sentence fits
        current_chunk_text += ' ' + sentence
        
        if len(tokenizer.encode(current_chunk_text)) >= chunk_size:

            # The chunk is full, finalize it
            if current_chunk_text:
                chunks.append(current_chunk_text.strip())

            # 2. Form the overlap for the new chunk from previous sentences
            overlap_sentences = []
            overlap_tokens = 0
            
            # Search backwards, taking sentences until at least <overlap> tokens are accumulated
            for s in reversed(current_sentences_in_chunk):
                s_tokens_count = len(tokenizer.encode(s))
                          
                overlap_tokens += s_tokens_count
                
                
                if overlap_tokens >= chunk_size * 0.8:
                    words = s.split()
                    overlap_tokens -= s_tokens_count
                    for w in reversed(words):
                        w_tokens_count = len(tokenizer.encode(w))
                        overlap_sentences.insert(0, w)
                        overlap_tokens += w_tokens_count
                        if overlap_tokens >= overlap:
                            break
                
                overlap_sentences.insert(0, s)

                # If the minimum overlap is reached, stop to minimize duplication.
                if overlap_tokens >= overlap:
                    break

            # 3. Start the new chunk with the overlap and add the current sentence
            
                     
            # Обновляем текущий текст чанка
            current_chunk_text = " ".join(overlap_sentences).strip()
            

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
            
    # logger.info(f'FLAGchunks {chunks}')
    return chunks, tail_sentences

 
