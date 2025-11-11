import re
import markdown
from bs4 import BeautifulSoup
import pypandoc
import nltk
from docx2python import docx2python
from io import BytesIO
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

def parse_pdf(content: bytes) -> list[tuple[str, int]]:
    """Parses a .pdf file and returns its text content, including OCR for images."""
    pdf_document = fitz.open(stream=content, filetype="pdf")
    text_by_page = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text("text")

        # OCR for images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            try:
                image = Image.open(BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image, lang='rus')
                page_text += "\n" + ocr_text
            except Exception as e:
                print(f"Error processing image on page {page_num + 1}: {e}")
    
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

def parse_txt(content: str) -> list[tuple[str, int | None]]:
    """Parses a .txt file and returns its text content."""
    return [(content, None)]

def clean_text(text: str) -> str:
    """Cleans the text by removing HTML/Markdown artifacts and normalizing whitespace."""
    cleaned_text = re.sub(r'<[^>]+>', ' ', text)
    cleaned_text = re.sub(r'&[a-z]+;|&#x?[0-9a-f]+;', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits the text into chunks of a specified size, respecting sentence boundaries."""
    sentences = nltk.sent_tokenize(text, language='russian')
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding the new sentence doesn't exceed the chunk size, add it
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            # If the current chunk is not empty, add it to the list
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start a new chunk with the current sentence. If the sentence itself
            # is larger than chunk_size, it will be in a chunk by itself.
            current_chunk = sentence

    # Add the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # Handle overlap - this is a simplified approach. A more robust solution
    # might involve adding sentences from the next chunk to the end of the current one.
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            # Get the last `overlap` characters from the previous chunk
            prev_chunk_overlap = chunks[i-1][-overlap:]
            overlapped_chunks.append(prev_chunk_overlap + chunks[i])
        return overlapped_chunks
    
    return chunks
