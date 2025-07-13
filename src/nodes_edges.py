from langgraph.types import Send
from typing import Literal
import json

from src.config import MAX_PROMPTS, REPROMPTING
from src.oifile import OIFile
from src.prompts import map_prompt, reduce_prompt
from src.states import OverallState, InputState, DetectState, LoadState, MaskCollectState, MaskState, SplitState
from src.utils import chunk_document, get_llm, get_logger


logger = get_logger()


async def _map_input(state: InputState) -> LoadState:
    """Map input files to load_document state."""
    sends = []

    # Send each file in the input state to the load_document state in parallel
    for file in state.get('files', []):
        sends.append(
            Send("load_document", {
                "file": file,
            })
        )

    return sends

async def _load_document(state: LoadState) -> OverallState:
    """Load a document from the provided file information dictionary."""
    results = []

    try:
        file = state.get('file', {})

        if file and isinstance(file, dict) and "file" in file:
            file_info = file["file"]
            logger.debug(f"Loading document with ID {file_info.get('id', '')}")

            if file_info.get('data', {}).get('content', ''):
                results.append(OIFile(
                    id=file_info['id'],
                    name=file_info['filename'],
                    type=file_info['meta']['content_type'],
                    content=file_info['data']['content']
                ))
                logger.debug(f"✓ Successfully loaded document: {results[0]}")
            else:
                logger.error(f"✕ ERROR: Missing data or content of document with ID {file_info.get('id', '')}")
        else:
            logger.warning(f"⚠ WARNING: Invalid document format received: {type(file)}")
    except Exception as e:
        logger.error(f"✕ ERROR: Could not load document: {str(e)}")

    return {'documents': results}

async def _map_documents_to_split(state: OverallState) -> SplitState:
    """Map loaded documents to split_document state."""
    sends = []

    for doc in state.get('documents', []):
        sends.append(
            Send("split_document", {
                "document": doc,
            })
        )

    return sends

async def _split_document(state: SplitState) -> OverallState:
    """Split a document into chunks."""
    results = {}

    file = state.get('document', None)

    if file:
        logger.debug(f"Splitting document: {file}")

        chunks = await chunk_document(file)

        if chunks:
            # Store chunks in dictionary with file ID as key
            results[file.get_id()] = list(chunks)

            logger.debug(f"✓ Successfully split document {file.get_name()} into {len(chunks)} chunks")
        else:
            logger.warning(f"⚠ WARNING: No chunks generated for document {file.get_name()}")
    else:
        logger.error("✕ ERROR: No document provided to '_split_document'")

    return {'document_chunks': results}

async def _map_chunks(state: OverallState) -> DetectState:
    """Map document chunks to identify_pii_items state."""
    sends = []

    n_prompts = state.get("n_prompts", 0)
    files_chunks = state.get('document_chunks', {})

    logger.debug(f"Map chunks received: {len(files_chunks)} documents")

    # For each document in the chunked_documents
    for file_id, chunks in files_chunks.items():
        logger.debug(f"Processing document {file_id} with {len(chunks)} chunks")

        # Log the content of the first chunk for debugging
        if chunks:
            logger.debug(f"First chunk sample: {chunks[0][:100]}...")

        # For each chunk in the document
        for content in chunks:
            # Create Send objects with document index metadata
            sends.append(
                Send("identify_pii_items",
                    {
                        "n_prompts": n_prompts,
                        "document_id": file_id,
                        "content": content,
                    }
                )
            )

    return sends

async def _identify_pii_items(state: DetectState) -> OverallState:
    """Identify PII items in the provided document content using an LLM."""
    results = {}

    n_prompts = state.get("n_prompts", 0)
    file_id = state.get("document_id", "")
    text = state.get("content", "")

    if file_id and text:
        prompt = await map_prompt.ainvoke({'text': text})
        llm = await get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content

        # Log the raw LLM response
        logger.debug(f"Raw LLM response for PII detection: {content}")

        if content.startswith("```json\n") and content.endswith("\n```"):
            content = content.replace("```json\n", "").replace("\n```", "").strip()

        # Also log the processed content
        logger.debug(f"Processed JSON content: {content}")

        results.update({
            "n_prompts": n_prompts + 1,
            "document_ids": [file_id],
            "partial_pii_items": [content],
        })

    return results

async def _group_pii_by_file(state: OverallState):
    """ Groups identified PII items by file ID for further processing."""
    results = {}

    # FIXED: Use document_ids instead of file_ids
    for file_id, items in zip(state.get("document_ids", []), state.get("partial_pii_items", [])):
        results.setdefault(file_id, []).append(items)

    logger.debug(f"Grouped PII by file: {results}")

    return {"document_partial_pii_items": results}

async def _should_reprompt(state: OverallState) -> Literal["mask_documents", "combine_file_pii_items"]:
    """ Determines whether to reprompt based on the current number of prompts."""
    logger.debug(f"N_PROMPTS: {state.get('n_prompts', 0)}")

    if REPROMPTING and state.get("n_prompts", 0) < MAX_PROMPTS:
        return "mask_documents"
    else:
        return "combine_file_pii_items"

async def _mask_documents(state: OverallState) -> OverallState:
    """
    Prepares document chunks and PII items for masking.

    Args:
        state: OverallState containing document chunks and PII items

    Returns:
        Updated state with masked chunks collection
    """
    # Initialize a collection for all chunks
    masked_chunks = {}

    files_chunks = state.get('document_chunks', {})

    if files_chunks:
        for file_id, chunks in files_chunks.items():
            # Start with copies of original chunks
            masked_chunks[file_id] = chunks.copy()

    return {"document_chunks": masked_chunks}

async def _route_masked_documents(state: OverallState) -> Literal["mask_text", "collect_masked_chunks"]:
    """
    Routes documents to either masking (if PII found) or directly to collection.
    """
    files_chunks = state.get('document_chunks', {})
    files_pii_items = state.get('document_partial_pii_items', {})

    # Check if any chunks need masking
    needs_masking = False
    if files_chunks and files_pii_items:
        for file_id, chunks in files_chunks.items():
            for chunk_idx, chunk in enumerate(chunks):
                if file_id in files_pii_items and chunk_idx < len(files_pii_items[file_id]):
                    pii_items = files_pii_items[file_id][chunk_idx]
                    if any(pii_item in chunk for pii_item in pii_items):
                        needs_masking = True
                        break
            if needs_masking:
                break

    # Route based on whether any chunks need masking
    if needs_masking:
        return "mask_text"
    else:
        return "collect_masked_chunks"

async def _mask_text(state: MaskState) -> OverallState:
    """
    Mask the content of a document based on previously identified PII items.
    This function replaces PII items with asterisks and updates the document collection.

    Args:
        state: MaskState containing document ID, content, and PII items

    Returns:
        Dictionary with the masked chunk and updated document chunks
    """
    results = {}

    file_id = state.get('document_id', "")
    text = state.get('content', '')
    pii_items = state.get('pii_items', [])
    chunk_index = state.get('chunk_index', 0)

    if file_id and text and pii_items and chunk_index is not None:
        # Create a new masked version of the text (fixing the replacement issue)
        masked_text = text
        for item in pii_items:
            masked_text = masked_text.replace(item, "*" * len(item))

        results.update({
            "masked_chunk": {
                "file_id": file_id,
                "chunk_index": chunk_index,
                "content": masked_text
            }
        })

    # Return with masked chunk that will update the document collection
    return results

async def _collect_masked_chunks(state: MaskCollectState):
    """ Collects masked document chunks and increments the n_prompts counter."""
    results = {}

    n_prompts = state.get('n_prompts', 0)  # Get current n_prompts value
    document_chunks = state.get('document_chunks', {})  # Use document_chunks instead of masked_document_chunks
    masked_chunk = state.get('masked_chunk', {})

    # If we have a new masked chunk, update our collection
    if document_chunks and masked_chunk:
        file_id = masked_chunk.get('file_id', '')
        chunk_index = masked_chunk.get('chunk_index', 0)
        content = masked_chunk.get('content', '')

        if file_id and file_id in document_chunks:
            # Update the specific chunk with masked content
            if chunk_index < len(document_chunks[file_id]):
                document_chunks[file_id][chunk_index] = content

        # Always set these values to ensure consistent state structure
        results.update({
            "document_chunks": document_chunks if document_chunks else {},
            "n_prompts": n_prompts + 1 if document_chunks and masked_chunk else n_prompts
        })

    return results

async def _combine_file_pii_items(state: OverallState):
    """ Combines identified PII items from different chunks of the same document into a single structured output."""
    results = {}

    for file_id, pii_items in state.get('document_partial_pii_items', {}).items():
        try:
            # FIXED: Safely handle JSON parsing
            pii_lists = []
            for pii_item in pii_items:
                try:
                    # Try to parse if it's a string
                    if isinstance(pii_item, str):
                        parsed_item = json.loads(pii_item)
                        pii_lists.append(json.dumps(parsed_item))
                    else:
                        # Already an object
                        pii_lists.append(json.dumps(pii_item))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse PII item: {pii_item[:100]}...")

            # Join the items for the prompt
            joined_lists = ',\n        '.join(pii_lists)

            # Generate the prompt and get LLM response
            prompt = await reduce_prompt.ainvoke({'pii_lists': joined_lists})
            llm = await get_llm()
            response = await llm.ainvoke(prompt)
            content = response.content

            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content.replace("```json\n", "").replace("\n```", "").strip()

            results[file_id] = content

            logger.debug(f"Combined PII for {file_id}: {content}")

        except Exception as e:
            logger.error(f"Error combining PII for file {file_id}: {str(e)}")

    return {'collected_pii_items': results}

async def _postprocess_pii_items(state: OverallState):
    '''
    Post-processes the collected PII items to remove duplicates based on 'text' and 'category' keys.
    '''
    results = []

    # Debug logging
    logger.debug(f"Postprocessing PII items from: {state}")

    # Access the correct key from the state
    collected_items = state.get('collected_pii_items', {})
    logger.debug(f"Collected items: {collected_items}")

    for file_id, pii_items in collected_items.items():
        try:
            # Parse JSON string to object
            logger.debug(f"Processing PII items for file {file_id}: {pii_items[:100]}...")

            # Handle both string and already parsed objects
            if isinstance(pii_items, str):
                parsed_items = json.loads(pii_items)
            else:
                parsed_items = pii_items

            logger.debug(f"Parsed items type: {type(parsed_items)}")

            # Remove duplicates based on 'text' and 'category' fields
            unique_items = set()
            unique_pii_items = []

            for item in parsed_items:
                # Create a hashable key from the 'text' and 'category' fields
                item_key = f"{item.get('text', '')}_{item.get('category', '')}"

                # Only add items we haven't seen before
                if item_key not in unique_items:
                    unique_items.add(item_key)
                    unique_pii_items.append(item)

            results.append(
                {
                    "id": file_id,
                    "pii": unique_pii_items,
                }
            )

            logger.debug(f"Added {len(unique_pii_items)} unique PII items for file {file_id}")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing PII items for file {file_id}: {e}")
            logger.error(f"Raw content: {pii_items}")
        except Exception as e:
            logger.error(f"Error processing PII items for file {file_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.debug(f"Final results: {results}")
    return {'final_pii_items': results}