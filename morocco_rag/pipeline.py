import os
import pickle
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .loader import MoroccoDocumentLoader
from .chunking import MultiRepresentationChunker
from .embeddings import get_embeddings
from .hybrid_retriever import HybridRetriever
from .reranker import CorrectiveReRanker

DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "morocco_vectorstore")

class MoroccoRAGPipeline:
    """
    Unified RAG Pipeline for Morocco AI, Digital Strategy, and Legal Documents.
    Prioritizes accurate document retrieval via:
      1. Dual-mode PDF extraction (PyMuPDF + Apple Vision OCR fallback)
      2. Multi-representation parent-child chunking (Notebook 12)
      3. Hybrid retrieval: Dense Semantic (Chroma) + Sparse Lexical (BM25)
      4. Reciprocal Rank Fusion (RRF k=60)
      5. Parent Context resolution (Notebook 12)
      6. Corrective RAG (CRAG) document relevance grading (Notebook 16)
    """
    def __init__(
        self,
        docs_dir: str = "Morroco",
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = "morocco_docs",
        embedding_provider: Optional[str] = None,
    ):
        self.docs_dir = docs_dir
        self.persist_dir = os.path.abspath(persist_dir)
        self.collection_name = collection_name
        self.embeddings = get_embeddings(embedding_provider)
        self.reranker = CorrectiveReRanker()
        self.vectorstore: Optional[Chroma] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.parent_store: Dict[str, Document] = {}
        self.child_docs: List[Document] = []

    def is_indexed(self) -> bool:
        """Check if persistent vectorstore and parent store exist."""
        parent_store_path = os.path.join(self.persist_dir, "parent_store.pkl")
        child_docs_path = os.path.join(self.persist_dir, "child_docs.pkl")
        return (
            os.path.exists(self.persist_dir)
            and os.path.exists(parent_store_path)
            and os.path.exists(child_docs_path)
        )

    def index(self, force_reindex: bool = False, max_docs: Optional[int] = None):
        """Build or load the persistent index."""
        os.makedirs(self.persist_dir, exist_ok=True)
        parent_store_path = os.path.join(self.persist_dir, "parent_store.pkl")
        child_docs_path = os.path.join(self.persist_dir, "child_docs.pkl")

        if self.is_indexed() and not force_reindex:
            print(f"Loading existing indexed vector store from: {self.persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            with open(parent_store_path, "rb") as f:
                self.parent_store = pickle.load(f)
            with open(child_docs_path, "rb") as f:
                self.child_docs = pickle.load(f)

            self.hybrid_retriever = HybridRetriever(
                vectorstore=self.vectorstore,
                parent_store=self.parent_store,
                child_docs=self.child_docs,
            )
            print(f"Index loaded. Child chunks: {len(self.child_docs)}, Parent contexts: {len(self.parent_store)}")
            return

        print("=== Starting Full Indexing of Morocco Documents ===")
        # 1. Load documents
        loader = MoroccoDocumentLoader(directory=self.docs_dir)
        raw_documents = loader.load_all(max_docs=max_docs)

        # 2. Chunk documents into parent-child hierarchy
        chunker = MultiRepresentationChunker()
        parent_docs, child_docs, parent_store = chunker.split_documents(raw_documents)
        print(f"\nChunking complete: {len(raw_documents)} raw pages -> {len(parent_docs)} parent chunks, {len(child_docs)} child chunks")

        # 3. Store child chunks in Chroma
        print(f"Generating embeddings and indexing {len(child_docs)} child chunks in Chroma...")
        self.vectorstore = Chroma.from_documents(
            documents=child_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )

        # 4. Save parent store and child docs cache
        self.parent_store = parent_store
        self.child_docs = child_docs
        with open(parent_store_path, "wb") as f:
            pickle.dump(parent_store, f)
        with open(child_docs_path, "wb") as f:
            pickle.dump(child_docs, f)

        # 5. Initialize Hybrid Retriever
        self.hybrid_retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            parent_store=self.parent_store,
            child_docs=self.child_docs,
        )
        print("=== Indexing Successfully Completed! ===\n")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_irrelevant: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top accurate documents for the given query.
        Combines Dense Semantic search + BM25 keyword search via RRF,
        resolves to full parent context, and performs CRAG relevance grading.
        """
        if not self.hybrid_retriever:
            self.index()

        # Step 1: Hybrid Retrieval with parent resolution (fetch 3x top_k candidates for grading)
        raw_candidates = self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k * 3,
            fetch_k=top_k * 5,
            resolve_to_parent=True,
        )

        # Step 2: Corrective RAG Re-ranking and Relevance Grading
        graded_results = self.reranker.grade_and_rerank(
            query=query,
            retrieved_items=raw_candidates,
            filter_irrelevant=filter_irrelevant,
        )

        return graded_results[:top_k]

    def format_context_for_generation(self, query: str, top_k: int = 5) -> str:
        """
        Formats retrieved documents into grounded context with exact citations
        ready to feed into an LLM generation step whenever desired.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documents found for this query in the Morocco knowledge base."

        context_blocks = []
        for i, item in enumerate(results, 1):
            source = item["source"]
            page = item["page"]
            total_pages = item["total_pages"]
            grade = item.get("crag_grade", "RELEVANT")
            confidence = item.get("confidence", 0.0)
            content = item["content"]

            block = (
                f"### [Document {i}] {source} (Page {page} of {total_pages})\n"
                f"- **Relevance Grade**: {grade} (Confidence: {confidence:.2f})\n"
                f"- **Matched Terms**: {', '.join(item.get('matched_terms', []))}\n"
                f"```text\n{content}\n```"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def as_tool(self):
        """Returns a LangChain tool compatible with create_agent."""
        from langchain_core.tools import tool

        pipeline = self

        @tool
        def search_morocco_documents(query: str) -> str:
            """
            Search Moroccan AI, cybersecurity, and digital transformation laws, strategies, and official reports.
            Returns precise citations with source document names, page numbers, and relevant text excerpts.
            """
            return pipeline.format_context_for_generation(query, top_k=5)

        return search_morocco_documents
