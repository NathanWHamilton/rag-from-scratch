import re
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

CROSS_LINGUAL_SYNONYMS = {
    "khwarizmi": ["khawarizmi"],
    "khawarizmi": ["khwarizmi"],
    "al-khwarizmi": ["al-khawarizmi"],
    "al-khawarizmi": ["al-khwarizmi"],
    "morocco": ["maroc", "moroccan", "royaume"],
    "maroc": ["morocco"],
    "law": ["loi", "dahir"],
    "loi": ["law"],
    "laws": ["lois"],
    "cybersecurity": ["cybersecurite", "cybersécurité"],
    "cybersecurite": ["cybersecurity"],
    "strategy": ["strategie", "stratégie"],
    "data": ["donnees", "données"],
    "protection": ["protection"],
    "personal": ["personnel", "personnelles"],
    "health": ["sante", "santé"],
    "education": ["enseignement", "formation"],
    "governance": ["gouvernance"],
}

def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizes text for BM25 while preserving:
    - Legal numbers & hyphenated codes (e.g. 09-08, 05-20)
    - Acronyms (e.g. CNDP, DGSSI, ADD, AI, UNESCO, UM6P)
    - Dollar / MAD figures and numbers (e.g. 100, 2030)
    """
    text = text.lower()
    # Match words, numbers, and hyphenated identifiers like '09-08' or 'loi-09-08'
    tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
    return tokens

class HybridRetriever:
    """
    Hybrid Retriever combining Dense Vector Search and BM25 Lexical Search
    using Reciprocal Rank Fusion (RRF) as detailed in rag-from-scratch notebooks 5, 6, 12, and 15.
    Resolves matched child chunks to their surrounding parent document context.
    """
    def __init__(
        self,
        vectorstore,
        parent_store: Dict[str, Document],
        child_docs: List[Document],
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        self.vectorstore = vectorstore
        self.parent_store = parent_store
        self.child_docs = child_docs
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Build BM25 index on child documents
        self.corpus_tokens = [tokenize_for_bm25(doc.page_content) for doc in child_docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search_dense(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """Retrieve top child chunks using dense semantic similarity with scores."""
        # Using similarity_search_with_score
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        # Note: Chroma score is cosine distance (lower is closer). We return doc and rank.
        return results

    def search_sparse(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """Retrieve top child chunks using BM25 with cross-lingual query expansion."""
        raw_tokens = tokenize_for_bm25(query)
        if not raw_tokens:
            return []
        expanded_tokens = list(raw_tokens)
        for t in raw_tokens:
            if t in CROSS_LINGUAL_SYNONYMS:
                expanded_tokens.extend(CROSS_LINGUAL_SYNONYMS[t])
        scores = self.bm25.get_scores(expanded_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.child_docs[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 25,
        resolve_to_parent: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. Fetches top candidate chunks from Dense search and BM25.
        2. Applies Reciprocal Rank Fusion (RRF) to merge the ranked lists.
        3. Resolves child chunks to parent chunks with contextual metadata.
        4. Returns deduplicated ranked results with detailed provenance.
        """
        dense_results = self.search_dense(query, top_k=fetch_k)
        sparse_results = self.search_sparse(query, top_k=fetch_k)

        # Reciprocal Rank Fusion score dictionary: doc_key -> { 'score': float, 'dense_rank': int, 'sparse_rank': int, 'doc': Document }
        rrf_scores: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense results
        for rank, (doc, dist) in enumerate(dense_results):
            child_id = doc.metadata.get("child_id") or doc.page_content[:60]
            if child_id not in rrf_scores:
                rrf_scores[child_id] = {
                    "doc": doc,
                    "rrf_score": 0.0,
                    "dense_rank": rank + 1,
                    "sparse_rank": None,
                    "dense_distance": dist,
                    "sparse_score": 0.0,
                }
            rrf_scores[child_id]["rrf_score"] += self.dense_weight * (1.0 / (self.rrf_k + rank + 1))

        # 2. Process Sparse results
        for rank, (doc, bm25_score) in enumerate(sparse_results):
            child_id = doc.metadata.get("child_id") or doc.page_content[:60]
            if child_id not in rrf_scores:
                rrf_scores[child_id] = {
                    "doc": doc,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": rank + 1,
                    "dense_distance": None,
                    "sparse_score": bm25_score,
                }
            else:
                rrf_scores[child_id]["sparse_rank"] = rank + 1
                rrf_scores[child_id]["sparse_score"] = bm25_score
            rrf_scores[child_id]["rrf_score"] += self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))

        # Sort all matched child chunks by RRF score descending
        sorted_candidates = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        # 3. Resolve to parent documents & deduplicate by parent_id
        final_results: List[Dict[str, Any]] = []
        seen_parents = set()

        for cand in sorted_candidates:
            child_doc = cand["doc"]
            parent_id = child_doc.metadata.get("parent_id")

            if resolve_to_parent and parent_id and parent_id in self.parent_store:
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)
                retrieved_doc = self.parent_store[parent_id]
                match_type = "Parent Context (via Child Match)"
            else:
                if child_doc.page_content in seen_parents:
                    continue
                seen_parents.add(child_doc.page_content)
                retrieved_doc = child_doc
                match_type = "Direct Chunk"

            result_entry = {
                "document": retrieved_doc,
                "content": retrieved_doc.page_content,
                "source": retrieved_doc.metadata.get("source", "Unknown"),
                "page": retrieved_doc.metadata.get("page", 1),
                "total_pages": retrieved_doc.metadata.get("total_pages", 1),
                "rrf_score": round(cand["rrf_score"], 6),
                "dense_rank": cand["dense_rank"],
                "sparse_rank": cand["sparse_rank"],
                "matched_child_snippet": child_doc.page_content[:200] + "...",
                "match_type": match_type,
            }
            final_results.append(result_entry)

            if len(final_results) >= top_k:
                break

        return final_results
