import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import markdown
from bs4 import BeautifulSoup
import pypandoc

def parse_docx(content: bytes) -> str:
    """Parses a .docx file and returns its text content."""
    return pypandoc.convert_text(content, 'plain', format='docx')

def parse_doc(content: bytes) -> str:
    """Parses a .doc file and returns its text content."""
    return pypandoc.convert_text(content, 'plain', format='doc')

def parse_rtf(content: bytes) -> str:
    """Parses an .rtf file and returns its text content."""
    return pypandoc.convert_text(content, 'plain', format='rtf')

def parse_md(content: str) -> str:
    """Parses a .md file and returns its text content."""
    html = markdown.markdown(content)
    return ''.join(BeautifulSoup(html, "html.parser").findAll(text=True))

def parse_txt(content: str) -> str:
    """Parses a .txt file and returns its text content."""
    return content

def clean_text(text: str) -> str:
    """Cleans the text by removing HTML/Markdown artifacts and normalizing whitespace."""
    cleaned_text = re.sub(r'<[^>]+>', ' ', text)
    cleaned_text = re.sub(r'&[a-z]+;|&#x?[0-9a-f]+;', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits the text into chunks of a specified size with a specified overlap."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=False
    )
    return splitter.split_text(text)
