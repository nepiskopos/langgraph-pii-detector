from langgraph.types import Send
from typing import Literal
import asyncio
import json

from src.config import MAX_PROMPTS, REPROMPTING
from src.oifile import OIFile
from src.prompts import map_prompt, reduce_prompt
from src.states import OverallState, InputState, DetectState
from src.utils import chunk_document, get_llm, get_logger, mask_content


logger = get_logger()


async def load_documents(state: InputState) -> dict[str, list[OIFile]]:
    '''
    Loads documents from the provided state, filtering for DOCX files.

    Args:
        state: InputState containing files to be processed

    Returns:
        A dictionary with a key "documents" containing a list of OIFile instances for each DOCX file found.
    '''
    body_files = state.get("files", [])

    if not body_files:
        logger.error("No files provided for processing.")
        raise ValueError("No files provided for processing.")

    logger.debug(f"Received {len(body_files)} files for processing")

    file_infos = []

    for entry in body_files:
        file = entry.get("file", {})

        # Only process DOCX files
        if file['filename'].casefold().endswith('.docx') and file['meta']['content_type'] == \
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # Create an OIFile instance and append it to the list
            file_infos.append(
                OIFile(file['id'], file['filename'], file['data']['content'])
            )

    return {"documents": file_infos}

async def split_documents(state: OverallState) -> dict[str, list[tuple[str]]]:
    '''
    Splits documents into chunks using the RecursiveCharacterTextSplitter.

    Args:
        state: OverallState containing documents to be chunked

    Returns:
        A dictionary with a key "chunked_documents" containing a list of tuples, each tuple
        representing the chunks of a document.
    '''
    documents = state.get("documents", [])
    n_prompts = state.get("n_prompts", 0)

    if not documents:
        logger.warning("No documents provided for chunking.")
        return {"chunked_documents": []}

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(25)

    if REPROMPTING and n_prompts > 0:
        logger.info(f"Reprompting requested. Masking documents contents before chunking.")

        file_ids = state.get("file_ids", [])
        pii_items = state.get("identified_pii_items", [])

        # Mask documents contents concurrently and gather results
        tasks = [mask_content(doc, file_ids, pii_items, semaphore) for doc in documents]
        documents = await asyncio.gather(*tasks)

    logger.debug(f"Chunking {len(documents)} documents")

    # Process documents concurrently and gather results
    tasks = [chunk_document(doc, semaphore) for doc in documents]
    chunked_documents = await asyncio.gather(*tasks)

    return {"chunked_documents": chunked_documents}

async def map_chunks(state: OverallState):
    '''
    Maps the chunked documents to the identify_pii_items function.

    This function creates Send objects for each chunk of each document,
    allowing the identify_pii_items function to process them in parallel.

    Args:
        state: OverallState containing chunked documents

    Returns:
        A list of Send objects, each containing a chunk of text and its associated file ID.
    '''
    # When used with add_conditional_edges, this function should return Send objects directly
    sends = []

    n_prompts = state.get("n_prompts", 0)
    files_ids = [doc.get_id() for doc in state.get("documents", [])]
    files_chunks = state.get('chunked_documents', [])

    # For each document in the chunked_documents (maintaining document separation)
    for file_id, chunks in zip(files_ids, files_chunks):
        # For each chunk in the document
        for content in chunks:
            # Create Send objects with document index metadata
            sends.append(
                Send("identify_pii_items",
                     {
                        "n_prompts": n_prompts,
                        "file_id": file_id,
                        "content": content,
                    }
                )
            )

    return sends  # Return Send objects directly without wrapping in a dictionary

async def identify_pii_items(state: DetectState):
    '''
    Identifies PII items in the provided text using a language model.

    This function takes the text content and file ID from the state,
    invokes the language model with a prompt, and returns the identified PII items.

    Args:
        state: DetectState containing the text content and file ID

    Returns:
        A dictionary with identified PII items and their associated file ID.

    This function is designed to be used with the map_chunks function,
    which provides the necessary text and file ID for processing.

    It processes each chunk of text independently, allowing for parallel PII detection.

    The function expects the text to be in a specific format (JSON) and returns the identified
    PII items in a structured format for further processing.

    If the content is not in the expected format, it will attempt to clean it up before returning.

    This function is asynchronous and designed to work within a larger state management system,
    allowing for efficient processing of multiple text chunks in parallel.
    '''
    n_prompts = state.get("n_prompts", 0)
    file_id = state.get("file_id", "")
    text = state.get("content", "")

    prompt = map_prompt.invoke({'text': text})
    llm = await get_llm()
    response = await llm.ainvoke(prompt)
    content = response.content

    if content.startswith("```json\n") and content.endswith("\n```"):
        content = content.replace("```json\n", "").replace("\n```", "").strip()

    # Return with document and chunk indexes to maintain structure
    return {
        "n_prompts": n_prompts + 1,
        "file_ids": [file_id],
        "identified_pii_items": [content],
    }

async def reprompt_to_identify_more_pii_items(state: OverallState) -> Literal["split_documents", "organize_pii_by_file"]:
    if REPROMPTING and state.get("n_prompts", 0) < MAX_PROMPTS:
        return "split_documents"
    else:
        return "organize_pii_by_file"

async def organize_pii_by_file(state: OverallState):
    '''
    Organizes identified PII items by their associated document IDs.

    This function takes the identified PII items and their corresponding file IDs
    from the state, and groups them by document ID.

    Args:
        state: OverallState containing identified PII items and file IDs

    Returns:
        A dictionary with a key "identified_pii_items_by_doc" containing a mapping of file IDs to their respective PII items.

    This function is used to structure the results of PII detection by document,
    allowing for easier aggregation and processing of PII items across multiple documents.

    It collects all identified PII items and groups them by their associated document IDs,
    enabling subsequent steps in the PII detection pipeline to work with a structured view of the data
    and to combine PII items from different chunks of the same document.

    This is particularly useful for scenarios where documents are split into chunks for processing,
    and the results need to be aggregated back into a coherent structure for further analysis or reporting.
    '''
    # Initialize structure to hold results by document
    results_by_doc = {}

    # Collect all items with their document indexes
    for file_id, items in zip(state.get("file_ids", []), state.get("identified_pii_items", [])):
        if file_id not in results_by_doc:
            results_by_doc[file_id] = []
        results_by_doc[file_id].append(items)

    return {"identified_pii_items_by_doc": results_by_doc}

async def combine_file_pii_items(state: OverallState):
    '''
    Combines identified PII items from different chunks of the same document into a single structured output.

    This function takes the identified PII items grouped by document ID and combines them into a single JSON structure
    for each document, ensuring that all PII items are aggregated correctly.

    Args:
        state: OverallState containing identified PII items by document

    Returns:
        A dictionary with a key "collected_pii_items" containing a mapping of file IDs to their respective combined PII items.

    This function is used to aggregate PII items from multiple chunks of the same document into a single coherent structure.

    It processes each document's identified PII items, invoking a language model to combine them into a structured JSON format.

    The resulting dictionary maps each file ID to its corresponding combined PII items,
    which can then be used for further processing or reporting.
    '''
    collected_pii_items = {}

    for file_id, pii_items in state['identified_pii_items_by_doc'].items():
        prompt = reduce_prompt.invoke({'pii_lists': ',\n        '.join([f'{json.loads(pii)}' for pii in pii_items])})
        llm = await get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content

        if content.startswith("```json\n") and content.endswith("\n```"):
            content = content.replace("```json\n", "").replace("\n```", "").strip()

        collected_pii_items[file_id] = content

    return {'collected_pii_items': collected_pii_items}

async def postprocess_pii_items(state: OverallState):
    '''
    Post-processes the collected PII items to remove duplicates based on 'text' and 'category' keys.

    This function takes the collected PII items from the state, parses them, and removes any duplicate entries
    based on the combination of 'text' and 'category' fields.

    Args:
        state: OverallState containing collected PII items
    Returns:
        A dictionary with a key "final_pii_items" containing a list of dictionaries, each representing a file ID and its unique PII items.

    This function is used to ensure that the final output of PII items is clean and free from duplicates,
    which is essential for accurate reporting and further processing of PII data.

    It processes each file's PII items, ensuring that only unique items are retained based on their 'text' and 'category' fields.

    The function efficiently handles the deduplication process by using a set to track unique items and creating a unique key for
    each PII item based on these fields, allowing for quick lookups and ensuring that the final output is concise and accurate.

    The function returns a structured output that can be easily consumed by downstream processes or reporting tools.

    This function is designed to be efficient and scalable, handling potentially large sets of PII items
    while ensuring that the final output is both unique and structured.

    The final output is structured as a list of dictionaries, where each dictionary contains the file ID and its corresponding
    unique PII items, ready for downstream processing or reporting.
    '''
    postprocessed_pii_items = []

    for file_id, pii_items in state['collected_pii_items'].items():
        pii_items = json.loads(pii_items)

        # Remove duplicates based on 'text' and 'category' fields
        unique_items = set()
        unique_pii_items = []

        for item in pii_items:
            # Create a hashable key from the 'text' and 'category' fields
            item_key = f"{item['text']}_{item['category']}"

            # Only add items we haven't seen before
            if item_key not in unique_items:
                unique_items.add(item_key)
                unique_pii_items.append(item)

        postprocessed_pii_items.append(
            {
                "id": file_id,
                "pii": unique_pii_items,
            }
        )

    return {'final_pii_items': postprocessed_pii_items}