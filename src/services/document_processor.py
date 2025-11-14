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

        # Define the area to extract text from, ignoring headers and footers
        # These values can be adjusted as needed
        header_margin = 50  # a small margin from the top
        footer_margin = page.rect.height - 50  # a small margin from the bottom

        # Define a clipping rectangle
        clip_rect = fitz.Rect(0, header_margin, page.rect.width, footer_margin)

        # Extract text within the defined rectangle
        page_text = page.get_text("text", clip=clip_rect)

        # Check for images and extract text using OCR
        image_list = page.get_images(full=True)
        if image_list:
            logger.info(f"Found {len(image_list)} images on page {page_num + 1}")
                            
            # for img_index, img in enumerate(image_list):
            #     try:
            #         xref = img[0]
            #         pix = fitz.Pixmap(pdf_document, xref)
                                    
            #         # Check if image is valid for OCR
            #         if pix.n - pix.alpha < 4:  # GRAY or RGB
            #             # Save image to temporary file for OCR processing
            #             temp_dir = tempfile.gettempdir()
            #             temp_img_path = os.path.join(temp_dir, f"pdf_image_{page_num}_{img_index}.png")
            #             pix.save(temp_img_path)
                        
            #             # Extract text using OCR
            #             ocr_text = pytesseract.image_to_string(temp_img_path, lang='rus')
            #             if ocr_text and len(ocr_text.strip()) > 10:  # Only add if meaningful text
            #                 page_text += "\n" + ocr_text
            #                 logger.info(f"OCR extracted {len(ocr_text)} characters from image on page {page_num + 1}")
            #             else:
            #                 logger.debug(f"OCR returned insufficient text from image on page {page_num + 1}")
                        
            #             # Clean up temporary file
            #             try:
            #                 os.remove(temp_img_path)
            #             except:
            #                 pass
                    
            #         pix = None

            #     except Exception as e:
            #         logger.info(f"Error processing image on page {page_num + 1}: {e}")
            #         pix = None
            #     finally:
            #             # Гарантирует, что файл не останется, даже если была ошибка.
            #             if os.path.exists(temp_img_path):
            #                 os.remove(temp_img_path)
        
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


# def parse_image(content: bytes) -> list[tuple[str, int | None]]:
#     """Parses an image file and returns its text content using OCR."""
#     try:
#         image = Image.open(BytesIO(content))
#         ocr_text = pytesseract.image_to_string(image, lang="rus")
#         return [(ocr_text, None)]
#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return []


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
    Splits the text into chunks of a specified token size with bidirectional overlap.
    The text is first split into paragraphs, which are then grouped into main content blocks.
    Each chunk is formed by taking a main block and prepending/appending paragraphs from
    neighboring blocks as contextual overlap.
    """
    if not text:
        return []

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
        return []

    # 2. Split long paragraphs into sentences
    docs_for_chunking = []
    for p in cleaned_paragraphs:
        p_tokens = tokenizer.encode(p)
        if len(p_tokens) > chunk_size:
            sentences = nltk.sent_tokenize(p, language="russian")
            docs_for_chunking.extend(sentences)
        else:
            docs_for_chunking.append(p)

    # 3. Create main content blocks (pre-chunks)
    main_blocks = []
    current_block_paras = []
    current_block_tokens = 0
    for para in docs_for_chunking:
        para_tokens = len(tokenizer.encode(para))
        if current_block_tokens + para_tokens > chunk_size and current_block_paras:
            main_blocks.append(current_block_paras)
            current_block_paras = []
            current_block_tokens = 0
        current_block_paras.append(para)
        current_block_tokens += para_tokens
    if current_block_paras:
        main_blocks.append(current_block_paras)

    # 4. Create final chunks with bidirectional overlap
    chunks = []
    for i, block in enumerate(main_blocks):
        # Add preceding overlap
        pre_overlap_paras = []
        if i > 0:
            prev_block = main_blocks[i-1]
            overlap_tokens = 0
            for p_rev in reversed(prev_block):
                p_rev_tokens = len(tokenizer.encode(p_rev))
                if overlap_tokens + p_rev_tokens <= overlap:
                    pre_overlap_paras.insert(0, p_rev)
                    overlap_tokens += p_rev_tokens
                else:
                    break

        # Add succeeding overlap
        post_overlap_paras = []
        if i < len(main_blocks) - 1:
            next_block = main_blocks[i+1]
            overlap_tokens = 0
            for p in next_block:
                p_tokens = len(tokenizer.encode(p))
                if overlap_tokens + p_tokens <= overlap:
                    post_overlap_paras.append(p)
                    overlap_tokens += p_tokens
                else:
                    break

        # Combine to form the final chunk
        final_chunk_paras = pre_overlap_paras + block + post_overlap_paras
        chunks.append("\n\n".join(final_chunk_paras))

    return chunks
