import uuid
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MultiRepresentationChunker:
    """
    Implements Multi-Representation Chunking (as demonstrated in rag_from_scratch_12_to_14.ipynb Part 12).
    Generates:
      1. Parent chunks: Larger chunks (e.g. 1500 chars) that preserve full context, legal clauses, and explanations.
      2. Child chunks: Smaller focused chunks (e.g. 450 chars) for high-accuracy embedding similarity.
    """
    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 450,
        child_chunk_overlap: int = 80,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", "Article ", "Chapitre ", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> Tuple[List[Document], List[Document], Dict[str, Document]]:
        """
        Takes raw page documents and creates:
        - parent_docs: List of parent documents with unique parent_id
        - child_docs: List of child documents linked to their parent_id
        - parent_store: Dict mapping parent_id -> parent Document
        """
        parent_docs: List[Document] = []
        child_docs: List[Document] = []
        parent_store: Dict[str, Document] = {}

        for doc in documents:
            parents = self.parent_splitter.split_documents([doc])
            for p_idx, parent in enumerate(parents):
                parent_id = str(uuid.uuid4())
                parent.metadata["parent_id"] = parent_id
                parent.metadata["parent_index"] = p_idx
                parent.metadata["is_parent"] = True
                parent_docs.append(parent)
                parent_store[parent_id] = parent

                # Split parent into children
                children = self.child_splitter.split_documents([parent])
                for c_idx, child in enumerate(children):
                    child_id = str(uuid.uuid4())
                    child.metadata["child_id"] = child_id
                    child.metadata["parent_id"] = parent_id
                    child.metadata["child_index"] = c_idx
                    child.metadata["is_child"] = True
                    # Preserve original metadata
                    child.metadata["source"] = parent.metadata.get("source")
                    child.metadata["page"] = parent.metadata.get("page")
                    child.metadata["total_pages"] = parent.metadata.get("total_pages")
                    child_docs.append(child)

        return parent_docs, child_docs, parent_store
