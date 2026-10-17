#!/usr/bin/env python3
import sys
import argparse
from morocco_rag import MoroccoRAGPipeline

def print_banner():
    print("=" * 75)
    print("   🇲🇦  MOROCCO ACCURATE RETRIEVAL RAG PIPELINE (HYBRID + CRAG)")
    print("=" * 75)

def format_result(item, index: int):
    source = item["source"]
    page = item["page"]
    total = item["total_pages"]
    confidence = item.get("confidence", 0.0)
    grade = item.get("crag_grade", "RELEVANT")
    dense_rank = item.get("dense_rank", "-")
    sparse_rank = item.get("sparse_rank", "-")
    matched_terms = item.get("matched_terms", [])
    content = item["content"].strip()

    print(f"\n[Result #{index}] 📄 {source} (Page {page} of {total})")
    print(f"  ├─ Grade:        {grade} (Score: {confidence:.3f})")
    print(f"  ├─ Provenance:   Dense Rank: #{dense_rank} | BM25 Rank: #{sparse_rank} | RRF Score: {item.get('rrf_score', 0):.5f}")
    if matched_terms:
        print(f"  ├─ Keyword Hits: {', '.join(matched_terms[:8])}")
    print(f"  └─ Excerpt:")
    # Print excerpt indented
    excerpt_lines = content[:600].split("\n")
    for line in excerpt_lines[:12]:
        if line.strip():
            print(f"     | {line[:95]}")
    if len(content) > 600:
        print("     | ... [truncated for display]")

def run_query(pipeline: MoroccoRAGPipeline, query: str, top_k: int = 5):
    print(f"\n🔍 Query: \"{query}\"")
    print("-" * 75)
    results = pipeline.retrieve(query, top_k=top_k)

    if not results:
        print("⚠️ No relevant documents found matching the confidence threshold.")
        return

    print(f"✅ Found {len(results)} high-precision verified passages:")
    for i, item in enumerate(results, 1):
        format_result(item, i)
    print("\n" + "=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Query Morocco RAG Document Knowledge Base")
    parser.add_argument("query", nargs="?", help="Question or query string to retrieve documents for")
    parser.add_argument("--top-k", type=int, default=4, help="Number of documents to retrieve (default: 4)")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild the vector & keyword index")
    args = parser.parse_args()

    print_banner()
    pipeline = MoroccoRAGPipeline(docs_dir="Morroco")

    if args.reindex or not pipeline.is_indexed():
        print("Building index from documents in Morroco/ ...")
        pipeline.index(force_reindex=args.reindex)

    if args.query:
        run_query(pipeline, args.query, top_k=args.top_k)
    else:
        print("\nEntering interactive retrieval mode (type 'exit' or 'quit' to stop):")
        while True:
            try:
                user_q = input("\nEnter query: ").strip()
                if not user_q or user_q.lower() in ("exit", "quit", "q"):
                    break
                run_query(pipeline, user_q, top_k=args.top_k)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting. Goodbye!")

if __name__ == "__main__":
    main()
