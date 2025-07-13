from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
import asyncio
import logging

from src.config import AZURE_OPENAI_MODEL_NAME, AZURE_OPENAI_API_VERSION, CHUNK_OVERLAP, CHUNK_SIZE
from src.oifile import OIFile


# Initialize global variables for logger and LLM
logger = None
llm = None


# Setup the AzureChatOpenAI LLM
async def get_llm():
    """Get the LLM for the langgraph agent."""
    global llm

    if not llm:
        llm = AzureChatOpenAI(
            model=AZURE_OPENAI_MODEL_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0.1,
        )
    return llm

def get_logger(name: str="pii-detector-map-reduce") -> logging.Logger:
    """
    Get a logger with the specified name. If no handlers are set, it will create a default StreamHandler.

    Args:
        name (str): The name of the logger. Defaults to "pii-detector-map-reduce".

    Returns:
        logging.Logger: The configured logger instance.
    """
    global logger

    if not logger:
        # Validate the logger name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Logger name must be a non-empty string.")

        # Get or create a logger with the specified name
        logger = logging.getLogger(name)

        # Ensure the logger is not already configured
        if not logger.hasHandlers():
            # If the logger does not have handlers, we will set it up
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.DEBUG)

    return logger

async def chunk_document(
    document: OIFile,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple:
    """Asynchronously chunk documents in parallel with limited concurrency"""
    logger = get_logger()

    # Create text splitter in a thread to avoid blocking
    def create_splitter_and_split_text(text: str) -> tuple:
        return tuple(RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", "... ", " ", ""],
            is_separator_regex=False
        ).split_text(text))

    # Validate parameters
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        logger.error(f"Invalid chunk parameters: size={chunk_size}, overlap={chunk_overlap}")
        raise ValueError(
            "Chunk size must be positive, overlap must be non-negative, and overlap must be less than chunk size."
        )

    if not document:
        logger.warning("No documents provided for chunking.")
        return ()

    name = document.get_name()
    text = document.get_content()

    if len(text) < chunk_size:
        logger.warning(f"Document '{name}' is shorter than chunk size. No chunking applied.")
        return tuple(text)

    # Initialize the return value
    split_docs = []

    logger.info(f"Chunking document '{name}' with length {len(text)} characters")

    # Process the text
    split_docs = await asyncio.to_thread(create_splitter_and_split_text, text)

    num_chunks = len(split_docs)
    if num_chunks == 0:
        logger.warning(f"No chunks created while splitting '{name}'")
    else:
        logger.info(f"Split '{name}' into {num_chunks} chunks")

    return tuple(split_docs)