import re

class OIFile:
    '''
    This is a class for representing a user-uploaded
    file object. It stores the file ID, the file name,
    the file content and the length of the fiel content.
    It also provides methods to access these attributes
    and to build the document content by normalizing
    newlines, replacing tabs with spaces, and collapsing
    multiple spaces into a single space.

    Attributes:
    - id (str): Unique identifier for the file.
    - name (str): Name of the file.
    - content (str): Normalized content of the file.
    - size (int): Size of the file content in bytes.
    '''
    def __init__(self, id: str, name: str, content: str):
        self.id = id
        self.name = name
        self.content = self._build_document(content)
        self.size = len(self.content)

    def __repr__(self) -> str:
        return f"File(id={self.id}, name={self.name}, size={self.size} bytes)"

    def _build_document(self, text: str) -> str:
        # First normalize consecutive newlines to single newlines
        document = re.sub(r'\n+', ' \n', text)

        # Replace tabs with spaces
        document = re.sub(r'\t', ' ', document)

        # Replace multiple consecutive spaces with a single space
        document = re.sub(r' +', ' ', document)

        return document

    def get_id(self) -> str:
        return self.id

    def get_name(self) -> str:
        return self.name

    def get_content(self) -> str:
        return self.content

    def get_size(self) -> int:
        return self.size

    def to_dict(self) -> dict:
        """
        Export the file object as a dictionary.
        """
        return {
            'id': self.id,
            'name': self.name,
            'content': self.content,
        }

    def update_content(self, content: str) -> None:
        """
        Update the content of the file and recalculate its size.
        """
        self.content = self._build_document(content)
        self.size = len(self.content)