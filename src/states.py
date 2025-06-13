from typing import Annotated, Any, TypedDict
import operator

from src.oifile import OIFile


class InputState(TypedDict):
    files: list[dict]

class OverallState(TypedDict):
    documents: list[OIFile]
    n_prompts: Annotated[int, lambda a, b: max(a, b)]
    chunked_documents: list[tuple[str]]
    file_ids: Annotated[list[str], operator.add]
    identified_pii_items: Annotated[list[str], operator.add]
    identified_pii_items_by_doc: dict[str, list[str]]
    collected_pii_items: dict[str, str]

class OutputState(TypedDict):
    final_pii_items: list[dict[str, Any]]

class DetectState(TypedDict):
    n_prompts: int
    file_id: str
    content: str