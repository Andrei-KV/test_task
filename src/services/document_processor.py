import re
import nltk
import tiktoken
from src.app.logging_config import get_logger
from unstructured.partition.auto import partition
from io import BytesIO

logger = get_logger(__name__)


def parse_document(content: bytes, file_name: str) -> list[tuple[str, int]]:
    """Parses a document and returns its text content."""
    elements = partition(file=BytesIO(content), file_filename=file_name)
    text_by_page = {}
    for el in elements:
        page_number = el.metadata.page_number or 1
        if page_number not in text_by_page:
            text_by_page[page_number] = ""
        text_by_page[page_number] += el.text + "\n"

    return sorted(text_by_page.items())


def clean_text(text: str) -> str:
    """Cleans the text by removing HTML/Markdown artifacts and normalizing whitespace."""
    cleaned_text = re.sub(r"<[^>]+>", " ", text)
    cleaned_text = re.sub(r"&[a-z]+;|&#x?[0-9a-f]+;", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    cleaned_text = re.sub(r"\n+", " ", cleaned_text).strip()
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
    pattern = r"\n\s*\n+"
    paragraphs = re.split(pattern, text)
    logger.info("paragraphs[:7]", paragraphs)
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
