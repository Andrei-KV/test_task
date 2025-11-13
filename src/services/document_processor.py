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


def parse_pdf(content: bytes) -> list[tuple[str, int, str | None]]:
    """
    Parses a .pdf file, extracts text, performs OCR on images, and identifies sections
    based on headers. Returns a list of tuples: (text_block, page_number, section_title).
    """
    pdf_document = fitz.open(stream=content, filetype="pdf")
    structured_text = []
    current_section = None

    # Regex to identify headers (e.g., "Глава 1", "Статья 2.3", "Пункт 3.1.1")
    section_pattern = re.compile(
        r"^\s*(Глава|Статья|Пункт|Раздел)\s*(\d+(\.\d+)*)\.?\s*(.*)", re.IGNORECASE
    )

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # Using blocks provides more structure than plain text
        blocks = page.get_text("blocks")

        for block in blocks:
            block_text = block[4].strip()

            # Check if the block is a section header
            match = section_pattern.match(block_text)
            if match:
                current_section = block_text
                # We can either skip adding the header as a separate block or include it.
                # Let's include it, it might contain useful context.

            if block_text:
                structured_text.append((block_text, page_num + 1, current_section))

        # OCR for images on the page
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            try:
                image = Image.open(BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image, lang='rus').strip()
                if ocr_text:
                    structured_text.append((ocr_text, page_num + 1, current_section))
            except Exception as e:
                print(f"Error processing image on page {page_num + 1}: {e}")
        
    return structured_text

def parse_docx(content: bytes) -> list[tuple[str, int, str | None]]:
    """
    Parses a .docx file, extracts text, and identifies sections based on headers.
    Returns a list of tuples: (text_block, page_number, section_title).
    """
    result = docx2python(BytesIO(content))
    structured_text = []
    current_section = None

    section_pattern = re.compile(
        r"^\s*(Глава|Статья|Пункт|Раздел)\s*(\d+(\.\d+)*)\.?\s*(.*)", re.IGNORECASE
    )

    for page_content in result.body:
        page_num = page_content.page_num
        for paragraph in page_content:
            paragraph_text = "".join(paragraph).strip()

            match = section_pattern.match(paragraph_text)
            if match:
                current_section = paragraph_text

            if paragraph_text:
                structured_text.append((paragraph_text, page_num, current_section))

    return structured_text

def parse_doc(content) -> list[tuple[str, int | None, str | None]]:
    """Parses a .doc file and returns its text content."""
    return [(pypandoc.convert_text(content, 'plain', format='doc'), None, None)]

def parse_rtf(content) -> list[tuple[str, int | None, str | None]]:
    """Parses an .rtf file and returns its text content."""
    return [(pypandoc.convert_text(content, 'plain', format='rtf'), None, None)]

def parse_md(content: str) -> list[tuple[str, int | None, str | None]]:
    # 1. Convert Markdown to HTML
    html = markdown.markdown(content)
    # 2. Parse the HTML
    soup = BeautifulSoup(html, "html.parser")
    text_content = soup.get_text(separator="\n", strip=True)
    return [(text_content, None, None)]

def parse_excel(content: bytes) -> list[tuple[str, int | None, str | None]]:
    """Parses an Excel file (.xls, .xlsx) and returns its text content."""
    xls = pd.ExcelFile(BytesIO(content))
    full_text = ""
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        full_text += df.to_string()
    return [(full_text, None, None)]

def parse_image(content: bytes) -> list[tuple[str, int | None, str | None]]:
    """Parses an image file and returns its text content using OCR."""
    try:
        image = Image.open(BytesIO(content))
        ocr_text = pytesseract.image_to_string(image, lang="rus")
        return [(ocr_text, None, None)]
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

def parse_txt(content: str) -> list[tuple[str, int | None, str | None]]:
    """Parses a .txt file and returns its text content."""
    return [(content, None, None)]


def clean_text(text: str) -> str:
    """Cleans the text by removing HTML/Markdown artifacts and normalizing whitespace."""
    cleaned_text = re.sub(r'<[^>]+>', ' ', text)
    cleaned_text = re.sub(r'&[a-z]+;|&#x?[0-9a-f]+;', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def split_text_into_chunks(
    structured_data: list[tuple[str, int, str | None]],
    chunk_size: int = 1000,
    overlap_sentences: int = 2,
) -> list[dict]:
    """
    Splits structured text data into chunks using a sliding window over sentences.
    Groups text by section and page to maintain context.
    """
    if not structured_data:
        return []

    tokenizer = tiktoken.get_encoding("cl100k_base")
    final_chunks = []

    # 1. Group text blocks by section and page to keep related content together
    sections = {}
    default_key = "Основной контент"
    for text, page, section_title in structured_data:
        # Group by a tuple of section and page to handle content correctly
        key = (section_title if section_title else default_key, page)
        if key not in sections:
            sections[key] = []
        sections[key].append(text)

    # 2. Process each group of text
    for (section_title, page), texts in sections.items():
        full_text = "\n\n".join(texts).strip()
        
        # 3. Split the text into sentences
        sentences = nltk.sent_tokenize(full_text, language="russian")

        if not sentences:
            continue

        # 4. Create chunks using a sliding window approach
        current_chunk_sentences = []
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence)
            
            # If the current chunk is full, store it and start a new one with an overlap
            if current_chunk_tokens + len(sentence_tokens) > chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                final_chunks.append({
                    "text": chunk_text,
                    "page": page,
                    "section": section_title if section_title != default_key else None
                })

                # Start the next chunk with an overlap of the last few sentences
                num_overlap = min(len(current_chunk_sentences), overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[-num_overlap:]
                current_chunk_tokens = len(tokenizer.encode(" ".join(current_chunk_sentences)))
            
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += len(sentence_tokens)

        # Add the final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            final_chunks.append({
                "text": chunk_text,
                "page": page,
                "section": section_title if section_title != default_key else None
            })

    return final_chunks
