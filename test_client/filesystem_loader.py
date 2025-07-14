from langchain_community.document_loaders import PyPDFLoader
from langchain_docling import DoclingLoader
from typing import Any, Dict, List, Optional
import asyncio
import mimetypes
import os
import pypandoc

from logger import get_logger


def load_local_documents(
    dir_path: str,
    recursive: bool = False,
    file_extensions: Optional[List[str]] = None,
    max_file_size_mb: Optional[float] = None,
    sort_by: str = "name",
    max_concurrency: int = 10
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for load_local_documents_async.
    """
    try:
        # Try to use the current event loop if running in async context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Event loop is already running")
        return loop.run_until_complete(
            load_local_documents_async(dir_path, recursive, file_extensions,
                                       max_file_size_mb, sort_by, max_concurrency)
        )
    except RuntimeError:
        # If no event loop exists or it's already running, create a new one
        return asyncio.run(
            load_local_documents_async(dir_path, recursive, file_extensions,
                                       max_file_size_mb, sort_by, max_concurrency)
        )

async def load_local_documents_async(
    dir_path: str,
    recursive: bool = False,
    file_extensions: Optional[List[str]] = None,
    max_file_size_mb: Optional[float] = None,
    sort_by: str = "name",  # Options: "name", "size", "modified"
    max_concurrency: int = 10
) -> List[Dict[str, Any]]:
    """
    Asynchronously load documents from a local directory,
    simulating the information structure of the files in Open WebUI.

    Args:
        dir_path (str): Path to the directory containing documents
        max_concurrency (int): Maximum number of files to process concurrently
        recursive (bool): Whether to scan subdirectories recursively
        file_extensions (List[str], optional): List of file extensions to include (e.g. ['.pdf', '.docx'])
        max_file_size_mb (float, optional): Maximum file size in MB to process
        sort_by (str): How to sort files before processing ("name", "size", or "modified")

    Returns:
        list: List of dictionaries containing document data
    """
    logger = get_logger("test_agent")

    # Resolve directory
    if not os.path.isabs(dir_path):
        dir_path = os.path.join(os.getcwd(), dir_path)

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        logger.warning(f"Directory not found: {dir_path}")
        return []

    # Find all files
    file_paths = []
    if recursive:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_paths.append(os.path.join(root, file))
    else:
        file_paths = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]

    # Apply extension filter if specified
    if file_extensions:
        file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in file_extensions]
        file_paths = [f for f in file_paths if os.path.splitext(f)[1].lower() in file_extensions]

    # Apply size filter if specified
    if max_file_size_mb:
        max_bytes = max_file_size_mb * 1024 * 1024
        file_paths = [f for f in file_paths if os.path.getsize(f) <= max_bytes]

    if not file_paths:
        logger.warning(f"No matching files found in directory: {dir_path}")
        return []

    # Sort files
    if sort_by == "name":
        file_paths.sort()
    elif sort_by == "size":
        file_paths.sort(key=os.path.getsize)
    elif sort_by == "modified":
        file_paths.sort(key=os.path.getmtime)

    # Define process file function with better error handling
    async def _process_file(idx: int, file_path: str, semaphore: asyncio.Semaphore):
        try:
            async with semaphore:
                logger.debug(f"Processing file {idx+1}/{len(file_paths)}: {file_path}")

                # Skip files that are too large
                if max_file_size_mb and os.path.getsize(file_path) > max_file_size_mb * 1024 * 1024:
                    logger.warning(f"Skipping file {file_path}: exceeds size limit of {max_file_size_mb}MB")
                    return None

                # Get document data with timeout protection
                try:
                    async with asyncio.timeout(30):  # 30 second timeout per file
                        doc_data = await get_document_data(file_path, file_id=str(idx))
                        return doc_data
                except asyncio.TimeoutError:
                    logger.error(f"Timeout while processing file {file_path}")
                    return None

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    # Process files in chunks to avoid memory issues with large directories
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []
    chunk_size = min(100, len(file_paths))  # Process up to 100 files at a time

    for i in range(0, len(file_paths), chunk_size):
        chunk = file_paths[i:i+chunk_size]
        tasks = [_process_file(i+j, path, semaphore) for j, path in enumerate(chunk)]
        chunk_results = await asyncio.gather(*tasks)
        results.extend([r for r in chunk_results if r])
        logger.debug(f"Processed {i+len(chunk)}/{len(file_paths)} files")

    logger.info(f"Successfully processed {len(results)}/{len(file_paths)} files from {dir_path}")
    return results

async def get_document_data(file_path: str, file_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get both metadata and content for a document file.

    Args:
        file_path (str): Path to the file
        file_id (str, optional): ID to assign to the file

    Returns:
        Dict[str, Any]: Dictionary containing file metadata and content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract content
    content = await extract_file_content(file_path)

    # Get metadata
    filename = os.path.basename(file_path)
    mime_type = await get_file_mime_type(file_path)
    file_size = os.path.getsize(file_path)

    # Simulate the information structure of the files in Open WebUI
    return {
        "file": {
            "id": file_id or os.path.splitext(filename)[0],
            "filename": filename,
            "meta": {
                "content_type": mime_type,
                "size": file_size
            },
            "data": {
                "content": content
            }
        }
    }

# Main function to extract file content based on type
async def extract_file_content(file_path: str) -> str:
    """
    Extract content from a file based on its extension.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Extracted text content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file extension in lowercase
    extension = os.path.splitext(file_path)[1].lower()

    # Map extensions to handler functions
    handlers = {
        '.txt': read_text_file,
        '.md': read_markdown_file,
        '.docx': read_docx_file,
        '.pdf': read_pdf_file,
        '.rtf': read_rtf_file,
        '.odt': read_odt_file
    }

    if extension in handlers:
        return await handlers[extension](file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

# File MIME type detection
async def get_file_mime_type(file_path: str) -> str:
    """
    Get the MIME type of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MIME type of the file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not mimetypes.inited:
        # Configure acceptable mimetypes
        mimetypes.init()
        mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')
        mimetypes.add_type('application/pdf', '.pdf')
        mimetypes.add_type('text/plain', '.txt')
        mimetypes.add_type('text/markdown', '.md')
        mimetypes.add_type('application/rtf', '.rtf')
        mimetypes.add_type('application/vnd.oasis.opendocument.text', '.odt')

    # Get mime type based on file extension
    mime_type, _ = mimetypes.guess_type(file_path)

    # Fallback if mime type couldn't be determined
    if not mime_type:
        # Check file signature
        file_start = await read_file_start(file_path)

        if file_start.startswith(b'%PDF'):
            return 'application/pdf'
        elif file_start.startswith(b'\x50\x4B\x03\x04'):  # ZIP signature (docx, xlsx)
            if file_path.endswith('.docx'):
                return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

        return 'application/octet-stream'

    return mime_type

async def read_file_start(file_path: str, bytes_to_read: int = 512) -> bytes:
    """Read the first bytes of a file to determine its type."""
    def _read(path: str, size: int) -> bytes:
        with open(path, 'rb') as f:
            return f.read(size)

    return await asyncio.to_thread(_read, file_path, bytes_to_read)

# Content readers for different file types
async def read_text_file(file_path: str) -> str:
    """Read content from a plain text file."""
    def _read(path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    return await asyncio.to_thread(_read, file_path)

async def read_markdown_file(file_path: str) -> str:
    """Read content from a markdown file."""
    # For basic markdown files, we can treat them as text
    return await read_text_file(file_path)

async def read_docx_file(file_path: str) -> str:
    """Extract text from DOCX file using DoclingLoader."""
    def _read_with_docling(path: str) -> str:
        documents = DoclingLoader(file_path=path).load()
        return '\n'.join([d.page_content for d in documents])

    return await asyncio.to_thread(_read_with_docling, file_path)

async def read_pdf_file(file_path: str) -> str:
    """Extract text from PDF file using PyPDFLoader."""
    def _read_with_pypdf(path: str) -> str:
        documents = PyPDFLoader(path).load()
        return '\n'.join([d.page_content for d in documents])

    return await asyncio.to_thread(_read_with_pypdf, file_path)

async def read_rtf_file(file_path: str) -> str:
    """Extract text from RTF file using pypandoc."""
    try:
        return await asyncio.to_thread(pypandoc.convert_file, file_path, "plain")
    except ImportError:
        raise ImportError("pypandoc is required for reading RTF files. Install it with 'pip install pypandoc'.")

async def read_odt_file(file_path: str) -> str:
    """Extract text from ODT file using pypandoc."""
    try:
        return await asyncio.to_thread(pypandoc.convert_file, file_path, "plain")
    except ImportError:
        raise ImportError("pypandoc is required for reading ODT files. Install it with 'pip install pypandoc'.")