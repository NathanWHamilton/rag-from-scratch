#!/usr/bin/env python3
"""
Runs a comprehensive benchmark test of 5 diverse questions on the Morocco RAG pipeline
and saves the full untruncated retrieval output to retrieval_test_results.md.
"""

import os
import sys
from datetime import datetime
from morocco_rag import MoroccoRAGPipeline

TEST_QUESTIONS = [
    {
        "id": 1,
        "category": "Economic Strategy & Targets",
        "question": "What are the GDP and job creation targets of Morocco's AI 2030 roadmap?",
    },
    {
        "id": 2,
        "category": "Data Privacy & Legal Framework",
        "question": "What does Moroccan Law No. 09-08 regulate and what is the role of CNDP?",
    },
    {
        "id": 3,
        "category": "Cybersecurity Legislation & Governance",
        "question": "What governance organs and crisis response committees are established under Cybersecurity Law No. 05-20?",
    },
    {
        "id": 4,
        "category": "Scientific Research & Applied AI",
        "question": "What is the Al-Khwarizmi program and which ministries partner in its execution?",
    },
    {
        "id": 5,
        "category": "Digital Transformation (OCR-scanned Strategy)",
        "question": "What are the strategic objectives and international ranking targets in the Digital Morocco 2030 strategy?",
    },
]

def generate_benchmark_report(output_file: str = "retrieval_test_results.md"):
    print("=" * 80)
    print("  RUNNING MOROCCO RAG 5-QUESTION BENCHMARK TEST (FULL OUTPUT)")
    print("=" * 80)

    pipeline = MoroccoRAGPipeline(docs_dir="Morroco")
    if not pipeline.is_indexed():
        print("Index not found. Indexing...")
        pipeline.index()

    markdown_lines = []
    markdown_lines.append("# Morocco RAG Pipeline: 5-Question Comprehensive Retrieval Test")
    markdown_lines.append(f"\n**Execution Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    markdown_lines.append("**Configuration:** Hybrid Retrieval (Dense BGE-Small + Sparse BM25) + Reciprocal Rank Fusion ($k=60$) + Corrective RAG (CRAG) Relevance Grading.")
    markdown_lines.append("**Corpus:** 38 Documents, 1,337 pages, 3,304 parent contexts, 11,200 child vectors in `Morroco/`.")
    markdown_lines.append("\n---\n")

    markdown_lines.append("## Executive Summary of Test Queries\n")
    markdown_lines.append("| # | Category | Query | Top Source Document | Top Match Grade |")
    markdown_lines.append("|---|---|---|---|---|")

    query_results = []

    for item in TEST_QUESTIONS:
        q_id = item["id"]
        cat = item["category"]
        query = item["question"]

        print(f"\n[{q_id}/5] Testing: '{query}' ({cat})")
        results = pipeline.retrieve(query, top_k=3, filter_irrelevant=False)
        query_results.append((item, results))

        if results:
            top = results[0]
            markdown_lines.append(f"| {q_id} | {cat} | {query} | `{top['source']}` (p. {top['page']}) | **{top.get('crag_grade', 'N/A')}** ({top.get('confidence', 0):.2f}) |")
        else:
            markdown_lines.append(f"| {q_id} | {cat} | {query} | *No results* | *N/A* |")

    markdown_lines.append("\n---\n")
    markdown_lines.append("## Detailed Results with Full Untruncated Excerpts\n")

    for item, results in query_results:
        q_id = item["id"]
        cat = item["category"]
        query = item["question"]

        markdown_lines.append(f"### Question {q_id}: {query}")
        markdown_lines.append(f"- **Focus Domain:** {cat}")
        markdown_lines.append(f"- **Total Candidate Passages Retrieved:** {len(results)}\n")

        for r_idx, doc_info in enumerate(results, 1):
            source = doc_info["source"]
            page = doc_info["page"]
            total_pages = doc_info["total_pages"]
            grade = doc_info.get("crag_grade", "RELEVANT")
            conf = doc_info.get("confidence", 0.0)
            rrf = doc_info.get("rrf_score", 0.0)
            dense_r = doc_info.get("dense_rank", "N/A")
            sparse_r = doc_info.get("sparse_rank", "N/A")
            matched_terms = doc_info.get("matched_terms", [])
            match_type = doc_info.get("match_type", "Context Chunk")
            content = doc_info["content"].strip()

            markdown_lines.append(f"#### Pass #{r_idx}: `{source}` — Page {page} of {total_pages}")
            markdown_lines.append(f"- **CRAG Relevance Grade:** `{grade}` (Confidence Score: `{conf:.4f}`)")
            markdown_lines.append(f"- **Retrieval Provenance:** Dense Rank: `#{dense_r}` | BM25 Rank: `#{sparse_r}` | Fused RRF: `{rrf:.6f}`")
            markdown_lines.append(f"- **Resolution Type:** {match_type}")
            if matched_terms:
                markdown_lines.append(f"- **Matched Query Keywords:** {', '.join(f'`{t}`' for t in matched_terms)}")
            markdown_lines.append("\n**Full Retrieved Document Text:**")
            markdown_lines.append("```text")
            markdown_lines.append(content)
            markdown_lines.append("```\n")

        markdown_lines.append("---\n")

    full_report = "\n".join(markdown_lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\n✅ Benchmark completed successfully! Report saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    generate_benchmark_report()
