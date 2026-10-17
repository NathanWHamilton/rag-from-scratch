import re
from typing import List, Dict, Any

STOPWORDS = {
    # English
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "d'", "l'", "ce", "cet", "cette", "ces", "dans",
    "en", "par", "pour", "sur", "avec", "sans", "sous", "et", "ou", "mais", "donc", "or", "ni", "car",
    "que", "qui", "quoi", "dont", "est", "sont", "ete", "être", "avoir", "au", "aux", "son", "sa", "ses"
}

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

def extract_meaningful_terms(query: str) -> List[str]:
    """Extract keywords, legal codes, and entity names from query."""
    tokens = re.findall(r'[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*', query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def term_matches(term: str, content: str) -> bool:
    """Check if term or any of its known transliterations/synonyms appear in content."""
    if term in content:
        return True
    for syn in CROSS_LINGUAL_SYNONYMS.get(term, []):
        if syn in content:
            return True
    return False

class CorrectiveReRanker:
    """
    Implements Re-ranking and CRAG Document Relevance Grading
    (from rag_from_scratch_15_to_18.ipynb Parts 15 & 16).
    Grades retrieved documents on relevance and suppresses irrelevant context.
    """
    def __init__(self, relevance_threshold: float = 0.25):
        self.relevance_threshold = relevance_threshold

    def grade_and_rerank(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        filter_irrelevant: bool = True
    ) -> List[Dict[str, Any]]:
        query_terms = extract_meaningful_terms(query)
        if not query_terms:
            return retrieved_items

        reranked = []
        for item in retrieved_items:
            content = item["content"].lower()
            
            # 1. Term hit count and coverage (with synonym/transliteration support)
            matched_terms = [term for term in query_terms if term_matches(term, content)]
            coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0

            # 2. Check for exact phrase or high-priority entities (numbers, laws like '09-08')
            special_tokens = [t for t in query_terms if any(c.isdigit() for c in t) or len(t) >= 6]
            special_hits = [t for t in special_tokens if term_matches(t, content)]
            special_boost = (len(special_hits) / len(special_tokens)) * 0.2 if special_tokens else 0.0

            # 3. Density: occurrences of query terms in the first 800 chars
            term_freq = sum(content.count(t) for t in query_terms)
            density_score = min(term_freq / (len(query_terms) * 4.0), 0.2)

            # Combined confidence score: Base RRF + coverage + special boost + density
            # Normalized between 0.0 and 1.0
            rrf_base = item.get("rrf_score", 0.0) * 10.0  # Scale RRF score (~0.015-0.035) to ~0.2-0.4
            confidence = min(round((coverage * 0.45) + rrf_base + special_boost + density_score, 4), 1.0)

            # Assign CRAG Grade
            if confidence >= 0.50:
                grade = "HIGH RELEVANCE"
            elif confidence >= self.relevance_threshold:
                grade = "MODERATE RELEVANCE"
            else:
                grade = "LOW RELEVANCE / NOISE"

            item_copy = dict(item)
            item_copy["confidence"] = confidence
            item_copy["crag_grade"] = grade
            item_copy["matched_terms"] = matched_terms
            item_copy["missing_terms"] = [t for t in query_terms if t not in matched_terms]

            if not filter_irrelevant or grade != "LOW RELEVANCE / NOISE" or len(retrieved_items) <= 1:
                reranked.append(item_copy)

        # Sort by final confidence descending
        reranked.sort(key=lambda x: x["confidence"], reverse=True)
        return reranked
