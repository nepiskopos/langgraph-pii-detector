from typing import Any
from pydantic_core import core_schema
import re
import html


class OIFile:
    '''
    This is a class for representing a user-uploaded document
    object. It stores the document ID, the document name, the
    document mimetype, the document content and the content summary.
    '''
    def __init__(self, id: str, name: str, type: str, content: str):
        self.id = id
        self.name = name
        self.type = type
        self.content = self._build_content(content)

    def get_id(self) -> str:
        return self.id

    def get_name(self) -> str:
        return self.name

    def get_type(self) -> str:
        return self.type

    def get_content(self) -> str:
        return self.content

    def get_size(self) -> int:
        return len(self.content)

    def set_content(self, content: str) -> None:
        self.content = content

    def _build_content(self, text_content: str) -> str:
        text = ''

        if text_content:
            text = text_content

            # Step 1: Remove HTML comments
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

            # Step 2: Remove HTML comments (duplicate step in original code)
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

            # Step 3: Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)

            # Step 4: Decode HTML entities like &nbsp;
            text = html.unescape(text)

            # Step 5: Fix spacing issues
            # Normalize multiple spaces
            text = re.sub(r' +', ' ', text)

            # Normalize newlines (no more than two consecutive)
            text = re.sub(r'\n{3,}', '\n\n', text)

            # Step 6: Fix specific layout issues from the document
            # Fix broken lines that should be together (like "Αριθμός Γ.Ε.ΜΗ .: 180526838000")
            text = re.sub(r'([a-zA-Zα-ωΑ-Ω])\.\s+:', r'\1.:', text)

            # Step 7: Remove extra spaces before punctuation
            text = re.sub(r' ([.,:])', r'\1', text)

            # Clean up trailing whitespace on each line
            text = '\n'.join(line.rstrip() for line in text.splitlines())

            # Clean up whitespaces at the beginning and ending of each string
            text.strip()

        return text

    def __repr__(self) -> str:
        return f"File(id={self.id}, name={self.name}, type={self.type}, size={len(self.content)} bytes)"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Tell Pydantic how to serialize/deserialize OIFile objects."""
        return core_schema.union_schema([
            # Handle OIFile instance
            core_schema.is_instance_schema(OIFile),
            # Convert dict to OIFile
            core_schema.chain_schema([
                core_schema.dict_schema(),
                core_schema.no_info_plain_validator_function(
                    lambda d: OIFile(
                        id=d.get("id"),
                        name=d.get("name"),
                        type=d.get("type"),
                        content=d.get("content"),
                    )
                ),
            ]),
        ])

    def to_dict(self) -> dict:
        """Convert OIFile instance to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "content": self.content,
        }