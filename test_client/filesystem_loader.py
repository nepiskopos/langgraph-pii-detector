import asyncio
import docx
import mimetypes
import os
import pypandoc
import time

from logger import get_logger


logger = get_logger()

def load_local_documents(dir_path: str, max_concurrency: int=10) -> list:
    '''
    Load local documents from a specified directory.

    This function reads all files in the given directory, extracts their content
    based on their type (text, DOCX, etc.), and returns a list of dictionaries
    containing file metadata and content.

    Args:
        dir_path (str): Path to the directory containing files to be processed.
        max_concurrency (int): Maximum number of files to process concurrently.

    Returns:
        list: List of dictionaries containing file metadata and content.

    This function will read all files in the specified directory, extract their content,
    and return a list of dictionaries with file metadata and content.

    If the directory does not exist or contains no files, it will return an empty list.

    If an error occurs while processing any file, it will log the error and continue processing
    the remaining files, returning a list of successfully processed files.

    If the directory path is relative, it resolves it to an absolute path based on the current
    working directory.

    If the directory does not exist or is not a directory, it returns an empty list.

    If no files are found in the directory, it logs a warning and returns an empty list.

    If the directory contains files, it processes them concurrently up to the specified limit
    using asyncio. It reads the content of each file, extracts metadata, and returns a list
    of dictionaries containing the file ID, name, and content
    '''
    # Resolve directory synchronously
    if not os.path.isabs(dir_path):
        dir_path = os.path.join(os.getcwd(), dir_path)

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return []

    # List files synchronously
    file_paths = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]

    if not file_paths:
        logger.warning(f"No files found in directory: {dir_path}")
        return []

    # Create async function to process files
    async def process_files():
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [_process_file(i, path, semaphore) for i, path in enumerate(file_paths)]
        return await asyncio.gather(*tasks)

    # Run the async function from sync code
    try:
        # Try to use the current event loop if running in async context
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(process_files())
    except RuntimeError:
        # If no event loop exists, create a new one
        results = asyncio.run(process_files())

    files = [result for result in results if result]
    logger.info(f"Successfully processed {len(files)}/{len(file_paths)} files")

    return files

async def _process_file(id: int, file_path: str, semaphore: asyncio.Semaphore) -> dict:
    """
    Process a single file with concurrency limit.
    This function reads the content of a file, extracts text based on its type,
    and returns an OIFile object containing the file ID, name, and content.
    If an error occurs during processing, it logs the error and returns None.

    Args:
        file_path (str): Path to the file to be processed.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.
    Returns:
        OIFile | None: OIFile object containing file details if successful, None if an error occurs.
    """
    async with semaphore:
        try:
            content = await _extract_file_content(file_path)
            file_info = await _extract_file_metadata(file_path)
            file_info["file"]["id"] = str(id)
            file_info["file"]["data"] = {"content": content}
            logger.info(f"Processed: {file_path}")
            return file_info
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {}

async def _extract_file_metadata(file_path: str) -> dict:
    """
    Extract metadata from a file.
    This function retrieves the file size and MIME type of the given file.

    Args:
        file_path (str): Path to the file to be processed.

    Returns:
        dict: Dictionary containing 'size' and 'mime_type' of the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    content_type = await _get_file_mime_type(file_path)
    created_at = int(time.time())

    return {
        "file": {
            "filename": filename,
            "created_at": created_at,
            "meta": {"content_type": content_type or "application/octet-stream"}
        }
    }

async def _extract_file_content(file_path: str) -> str:
    """
    Extract text from file based on extension.
    This function reads the content of a file based on its type (text, DOCX, etc.)
    and returns the text content. It supports text files, DOCX files, and special formats
    handled by pypandoc. If the file type is unsupported, it raises a ValueError.

    Args:
        file_path (str): Path to the file to be processed.

    Returns:
        str: Extracted text content from the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = os.path.splitext(file_path)[1].casefold()

    # File type handlers
    handlers = {
        ".txt": _read_text_file,
        ".docx": _read_docx_file,
        # ".pdf": lambda path: _process_pdf_file(path),  # Keep async
    }

    # Special case for pypandoc formats
    if extension in [".rtf", ".odt", ".md"]:
        return await asyncio.to_thread(pypandoc.convert_file, file_path, "plain")

    # Use handler if available
    if extension in handlers:
        handler = handlers[extension]
        # All our handlers are async functions
        return await handler(file_path)

    # Unsupported format
    raise ValueError(f"Unsupported file format: {extension}")

async def _read_text_file(file_path: str) -> str:
    """Read and return text file content."""
    return await asyncio.to_thread(lambda path: open(path, "r", encoding="utf-8").read(), file_path)

async def _read_docx_file(file_path: str) -> str:
    """Extract text from DOCX file."""
    def read_docx(path):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    return await asyncio.to_thread(read_docx, file_path)

# async def _process_pdf_file(file_path: str) -> str:
#     """Process PDF using Azure Document Intelligence."""
#     doc_client = DocumentIntelligenceService()
#     result = await asyncio.to_thread(doc_client.analyze_document, file_path)
#     return await asyncio.to_thread(doc_client.post_process_results, result)

async def _get_file_mime_type(file_path: str) -> str:
    """
    Asynchronously get the MIME type of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MIME type of the file, or 'application/octet-stream' if unknown
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use mimetypes library to guess type based on extension
    mime_type, _ = await asyncio.to_thread(mimetypes.guess_type, file_path)

    # If mime_type is None, try to determine by reading a bit of the file
    if mime_type is None:
        # This is a simplified approach - for better results, consider python-magic library
        file_start = await _read_file_start(file_path)

        # Very basic type detection based on file signatures
        if file_start.startswith(b'%PDF'):
            return 'application/pdf'
        elif file_start.startswith(b'\x50\x4B\x03\x04'):  # ZIP signature (could be docx, xlsx, etc.)
            return 'application/zip'

        # Default to binary
        return 'application/octet-stream'

    return mime_type

async def _read_file_start(path: str, bytes_to_read: int=512) -> bytes:
    """Read first few bytes to help determine file type"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'rb') as f:
        return f.read(bytes_to_read)