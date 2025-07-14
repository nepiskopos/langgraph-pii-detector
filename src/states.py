from typing import Annotated, Any, Dict, List, TypedDict
import operator

from src.oifile import OIFile


class InputState(TypedDict):
    """Input state for the PII detection process, containing the documents to be processed."""
    files: List[dict]

class OverallState(TypedDict):
    """Overall state for the PII detection process, containing all necessary information."""
    n_prompts: Annotated[int, lambda a, b: max(a, b)]
    documents: Annotated[List[OIFile], operator.add]
    document_chunks: Annotated[Dict[str, List[str]], operator.or_]
    document_ids: Annotated[List[str], operator.add]
    partial_pii_items: Annotated[List[str], operator.add]
    document_partial_pii_items: Dict[str, List[str]]
    collected_pii_items: Annotated[Dict[str, str], operator.or_]
    masked_chunks: Annotated[List[Dict[str, str]], operator.add]

class OutputState(TypedDict):
    """Output state for the PII detection process, containing the final detected PII items."""
    final_pii_items: List[dict[str, Any]]


class LoadState(TypedDict):
    """State for the load node that contains a documents information to be loaded as an IOFile object."""
    file: Dict[str, Any]

class SplitState(TypedDict):
    """State for the split node that contains an IOFile object whose content will be split into chunks."""
    document: OIFile

class DetectState(TypedDict):
    """State for the detect node that contains a document and a list of PII items to detect in the document content."""
    n_prompts: int
    document_id: str
    content: str

class MaskState(TypedDict):
    """State for the mask node that contains an document and a list of PII items to mask in the document content."""
    document_id: str
    chunk_index: int
    chunk_content: str
    pii_items: List[str]

class ReduceState(TypedDict):
    """State for the reduce node that contains a list of PII items to be combined."""
    document_id: str
    partial_pii_items: List[str]
    collected_pii_items: Dict[str, str]