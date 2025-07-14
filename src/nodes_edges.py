from langgraph.types import Send
from typing import Literal
import json

from src.config import MAX_PROMPTS, REPROMPTING
from src.oifile import OIFile
from src.prompts import map_prompt, reduce_prompt
from src.states import OverallState, InputState, OutputState, DetectState, LoadState, MaskState, ReduceState, SplitState
from src.utils import chunk_document, get_llm, get_logger, mask_text_with_normalization, pii_exist_in_text


logger = get_logger()


async def _map_input(state: InputState) -> LoadState:
    """Map input files to load_document state."""
    sends = []

    # Send each document in the input state to the load_document state in parallel
    for file in state.get('files', []):
        sends.append(
            Send("load_document", {
                "file": file,
            })
        )

    return sends

async def _load_document(state: LoadState) -> OverallState:
    """Load a document from the provided document information dictionary."""
    results = []

    try:
        file = state.get('file', {})

        if file and isinstance(file, dict) and "file" in file:
            file_info = file["file"]

            logger.debug(f"Loading document {file_info.get('filename', '')}...")

            if file_info.get('data', {}).get('content', ''):
                results.append(OIFile(
                    id=file_info['id'],
                    name=file_info['filename'],
                    type=file_info['meta']['content_type'],
                    content=file_info['data']['content']
                ))
                logger.info(f"✓ Successfully loaded document: {results[0]}")
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
        logger.debug(f"Splitting document {file.get_name()}")

        chunks = await chunk_document(file)

        if chunks:
            # Store chunks in dictionary with document ID as key
            results[file.get_id()] = list(chunks)

            logger.info(f"✓ Successfully split document {file.get_name()} into {len(chunks)} chunks")
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

    logger.debug(f"_map_chunks: Received {len(files_chunks)} documents")

    for file_id, chunks in files_chunks.items():
        if chunks:
            for content in chunks:
                if content.strip():
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
        try:
            logger.debug(f"Identifying PII items in a chunk from document {file_id} with content {text[:100]}...")

            prompt = await map_prompt.ainvoke({'text': text})
            llm = await get_llm()
            response = await llm.ainvoke(prompt)
            content = response.content

            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content.replace("```json\n", "").replace("\n```", "").strip()

                pii = json.loads(content)

                if pii and isinstance(pii, list):
                    results.update({
                        "n_prompts": n_prompts + 1,
                        "document_ids": [file_id],
                        "partial_pii_items": [content],
                    })

                    logger.info(f"Identified {len(pii)} PII items in a chunk from document with ID {file_id}.")
        except json.JSONDecodeError as e:
            logger.warning(f"No PII items identified in a chunk from document {file_id} with content {text[:100]}")
        except Exception as e:
                logger.error(f"Failed to identify PII items in a chunk from a document with ID {file_id}.")

    return results

async def _group_pii_by_file(state: OverallState) -> OverallState:
    """ Groups identified PII items by document ID for further processing."""
    results = {}

    logger.debug(f"Grouping partially identified PII items by document IDs...")

    for file_id, items in zip(state.get("document_ids", []), state.get("partial_pii_items", [])):
        results.setdefault(file_id, []).append(items)

    logger.info(f"Grouped partially identified PII items by document IDs: {results}")

    return {"document_partial_pii_items": results}

async def _map_file_partial_pii(state: OverallState) -> ReduceState:
    sends = []

    for file_id, pii_items in state.get('document_partial_pii_items', {}).items():
        sends.append(
            Send("combine_file_pii_items", {
                "document_id": file_id,
                "partial_pii_items": pii_items,
                "collected_pii_items": state.get("collected_pii_items", {}),
            })
        )

    return sends

async def _combine_file_pii_items(state: ReduceState) -> OverallState:
    """ Combines identified PII items from different chunks of the same document into a single structured output."""
    results = {}

    file_id = state.get("document_id", "")
    pii_items = state.get("partial_pii_items", [])
    existing_file_pii = state.get("collected_pii_items", {})

    if file_id and pii_items:
        logger.debug(f"Combining partially identified PII items for document with ID {file_id} into a final list of identified PII...")

        try:
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
            joined_lists = '\n'.join(f"- {item}" for item in pii_lists)

            # Generate the prompt and get LLM response
            prompt = await reduce_prompt.ainvoke({'pii_lists': joined_lists})
            llm = await get_llm()
            response = await llm.ainvoke(prompt)
            content = response.content

            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content.replace("```json\n", "").replace("\n```", "").strip()

            pii = json.loads(content)

            if pii and isinstance(pii, list):
                combined_pii = json.loads(existing_file_pii.get(file_id, '[]')) + pii

                results[file_id] = json.dumps(combined_pii)

                logger.info(f"Combined identified PII for document with ID {file_id} into a list of {len(combined_pii)} PII items")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to combine identified PII for document with ID {file_id} into a list of PII items")
        except Exception as e:
            logger.error(f"Failed to combine identified PII for document with ID {file_id} into a list of PII items")

    return {'collected_pii_items': results}

async def _should_reprompt(state: OverallState) -> Literal["mask_documents", "postprocess_pii_items"]:
    """ Determines whether to reprompt based on the current number of prompts."""
    if REPROMPTING and state.get("n_prompts", 0) < MAX_PROMPTS:
        logger.info(f"Performed {state.get('n_prompts', 0)} out of {MAX_PROMPTS} requested prompts --> Reprompting for identifying more PII items.")
        return "mask_documents"
    else:
        logger.info(f"Reached the maximum number of allowed prompts: {state.get('n_prompts', 0)} out of {MAX_PROMPTS}.")
        return "postprocess_pii_items"

async def _mask_documents(state: OverallState) -> OverallState:
    """
    Prepares document chunks and PII items for masking.

    Args:
        state: OverallState containing document chunks and PII items

    Returns:
        Updated state with masked chunks collection
    """
    results = {}

    files_chunks = state.get('document_chunks', {})
    files_pii_items = state.get('document_partial_pii_items', {})

    if files_chunks and files_pii_items:
        file_ids = set(files_chunks.keys()).union(set(files_pii_items.keys()))

        for file_id in file_ids:
            # Prepare chunks for the current file
            chunks = files_chunks.get(file_id, [])

            # Get PII items for the file, defaulting to an empty list if not found
            pii_items = files_pii_items.get(file_id, [])

            if chunks and pii_items:
                pii_list = [item['text'] for item in json.loads(pii_items[0])]

                for chunk_idx, chunk in enumerate(chunks):
                    needs_masking = await pii_exist_in_text(chunk, pii_list)

                    if needs_masking:
                        results.setdefault(file_id, []).append(chunk)

                        logger.debug(f"Chunk {chunk_idx+1} of document {file_id} requires masking before repromting.")

    return {"document_chunks": results}

async def _map_masked_chunks(state: OverallState) -> MaskState:
    """Map masked chunks to the mask_text state."""
    sends = []

    document_chunks = state.get('document_chunks', {})

    for file_id, chunks in document_chunks.items():
        for chunk_index, chunk_content in enumerate(chunks):
            sends.append(
                Send("mask_text", {
                    "document_id": file_id,
                    "chunk_index": chunk_index,
                    "chunk_content": chunk_content,
                    "pii_items": state.get('document_partial_pii_items', {}).get(file_id, [])
                })
            )

    return sends

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

    file_id = state.get('document_id', '')
    chunk_index = state.get('chunk_index', 0)
    text = state.get('chunk_content', '')
    pii_items = state.get('pii_items', [])

    if file_id and text and pii_items:
        pii_list = [item['text'] for item in json.loads(pii_items[0])]

        logger.debug(f"Masking chunk {chunk_index} of document {file_id}: {text[:100]}...")

        # Create a new masked version of the text (fixing the replacement issue)
        masked_text = await mask_text_with_normalization(text, pii_list)

        results.update({
            "masked_chunks": [{
                "file_id": file_id,
                "chunk_index": chunk_index,
                "content": masked_text
            }]
        })

        logger.debug(f"Masked chunk {chunk_index} of document {file_id}: {masked_text[:100]}...")

    # Return with masked chunk that will update the document collection
    return results

async def _collect_masked_chunks(state: OverallState) -> OverallState:
    """ Collects masked document chunks and increments the n_prompts counter."""
    results = {}

    document_chunks = state.get('document_chunks', {})
    masked_chunks = state.get('masked_chunks', [])

    # If we have a new masked chunk, update our collection
    if document_chunks and masked_chunks:
        logger.debug(f"Grouping masked document chunks by document ID...")

        for file_id in document_chunks.keys():
            for entry in masked_chunks:
                if file_id == entry.get('file_id', ''):
                    document_chunks[file_id][entry.get('chunk_index', 0)] = entry.get('content', '')

        # Always set these values to ensure consistent state structure
        results.update({
            "document_chunks": document_chunks,
        })

        logger.info(f"Grouped masked document chunks by document ID...")

    return results

async def _postprocess_pii_items(state: OverallState) -> OutputState:
    '''
    Post-processes the collected PII items to remove duplicates based on 'text' and 'category' keys.
    '''
    results = []

    # Access the correct key from the state
    collected_items = state.get('collected_pii_items', {})
    logger.debug(f"Collected items: {collected_items}")

    for file_id, pii_items in collected_items.items():
        try:
            logger.debug(f"Postprocessing final PII items for document with ID {file_id}...")

            # Handle both string and already parsed objects
            if isinstance(pii_items, str):
                parsed_items = json.loads(pii_items)
            else:
                parsed_items = pii_items

            logger.debug(f"Parsed PII items type: {type(parsed_items)}")

            # Remove duplicates based on 'text' and 'category' fields
            unique_items = set()
            unique_pii_items = []

            for item in parsed_items:
                # Create a hashable key from the 'text' and 'category' fields
                item_key = f"{item.get('text', '')}_{item.get('category', '')}"

                # Only add items we haven't seen before
                if item_key not in unique_items:
                    if '****' not in item.get('text', ''):
                        # Exclude masked and empty items
                        unique_items.add(item_key)
                        unique_pii_items.append(item)

            results.append(
                {
                    "id": file_id,
                    "pii": unique_pii_items,
                }
            )

            logger.info(f"Identified {len(unique_pii_items)} valid unique PII items for document {file_id}")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing PII items for document {file_id}: {e}")
        except Exception as e:
            logger.error(f"Error processing PII items for document {file_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.debug(f"Final PII result: {results}")

    return {'final_pii_items': results}