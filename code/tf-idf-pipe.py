"""
TF-IDF Pipeline for Toxicity Scoring (memory-efficient)
========================================================
Uses generators to stream the CSV — never holds the full dataset in RAM.
Two passes over the file:
  Pass 1: build df_counts (term → doc frequency) and count N
  Pass 2: compute TF-IDF on demand via tfidf_score(term, doc_index)
"""

import re
import math
import csv
from collections import defaultdict


CSV_PATH = "../data/train.csv"
MIN_DOC_FREQ = 3   # ignore terms that appear in fewer than this many docs


# ── 1. Text preprocessing ─────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "it", "this", "that", "be", "was",
    "are", "were", "i", "you", "he", "she", "we", "they", "so",
    "do", "my", "your", "not", "will", "have", "has", "had",
}

def preprocess(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


# ── 2. Generator: stream rows from CSV ───────────────────────────────────────
# Yields (doc_index, comment_text, target) one row at a time — no full load

def stream_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            yield i, row.get("comment_text", ""), row.get("target", "0")


# ── 3. Pass 1 — build df_counts and N ────────────────────────────────────────
# Streams the entire CSV once, keeping only a {term: int} counter in memory

print("Pass 1: building document frequency counts...")

df_counts: dict[str, int] = defaultdict(int)
N = 0

for doc_index, text, _ in stream_csv(CSV_PATH):
    tokens = set(preprocess(text))   # set = count each term once per doc
    for term in tokens:
        df_counts[term] += 1
    N += 1
    if N % 500_000 == 0:
        print(f"  {N:,} docs processed, vocab size: {len(df_counts):,}")

# Filter out rare terms (saves memory and reduces noise)
df_counts = {t: c for t, c in df_counts.items() if c >= MIN_DOC_FREQ}
print(f"Done. N={N:,} documents, vocabulary={len(df_counts):,} terms (min_df={MIN_DOC_FREQ})\n")


# ── 4. IDF (computed once from df_counts) ────────────────────────────────────

def idf(term: str) -> float:
    """Smoothed IDF: log((N+1) / (df+1)) + 1"""
    doc_freq = df_counts.get(term, 0)
    return math.log((N + 1) / (doc_freq + 1)) + 1


# ── 5. Pass 2 — fetch a single document's TF dict on demand ──────────────────
# Streams to the target row and stops — no random access needed

def _get_doc_tf(doc_index: int) -> dict[str, int]:
    """Stream to doc_index and return its {term: count} dict."""
    for i, text, _ in stream_csv(CSV_PATH):
        if i == doc_index:
            tokens = preprocess(text)
            counts: dict[str, int] = defaultdict(int)
            for t in tokens:
                counts[t] += 1
            return dict(counts)
    raise IndexError(f"doc_index {doc_index} out of range (N={N})")


# ── 6. TF-IDF score function ──────────────────────────────────────────────────

def tfidf_score(term: str, doc_index: int) -> float:
    """
    Returns the TF-IDF score for a term in a document.

    Parameters
    ----------
    term      : word to score (normalised internally)
    doc_index : 0-based row index in the CSV

    Returns
    -------
    float — 0.0 if the term is not present in the document
    """
    term = term.lower().strip()
    tf = _get_doc_tf(doc_index)
    term_count = tf.get(term, 0)
    doc_length = sum(tf.values()) or 1
    normalised_tf = term_count / doc_length
    return normalised_tf * idf(term)


# ── 7. TF-IDF vector for a document ──────────────────────────────────────────

def tfidf_vector(doc_index: int) -> dict[str, float]:
    """Returns {term: tfidf_score} for every known term in the document."""
    tf = _get_doc_tf(doc_index)
    doc_length = sum(tf.values()) or 1
    return {
        term: (count / doc_length) * idf(term)
        for term, count in tf.items()
        if term in df_counts          # skip terms filtered out by min_df
    }


# ── 8. Batch vector generator (for training) ─────────────────────────────────
# Use this instead of calling tfidf_vector() in a loop — single streaming pass

def stream_tfidf_vectors(vocab: list[str] | None = None):
    """
    Yields (doc_index, target, tfidf_dict) for every row.
    Streams the CSV once — use this to build your training matrix.

    Parameters
    ----------
    vocab : optional fixed vocabulary list; if given, only those terms are scored
    """
    vocab_set = set(vocab) if vocab else None

    for doc_index, text, target in stream_csv(CSV_PATH):
        tokens = preprocess(text)
        counts: dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        doc_length = sum(counts.values()) or 1

        vec = {
            term: (count / doc_length) * idf(term)
            for term, count in counts.items()
            if term in df_counts and (vocab_set is None or term in vocab_set)
        }
        yield doc_index, float(target or 0), vec


# ── 9. Quick demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Pass 2: sampling first 10 documents...\n")

    for doc_index, target, vec in stream_tfidf_vectors():
        if doc_index >= 10:
            break
        top3 = sorted(vec.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"[{doc_index}] toxicity={target:.2f}")
        for term, score in top3:
            print(f"       {term:<20} TF-IDF = {score:.4f}")
        print()

    print("─" * 50)
    print("Single lookup: tfidf_score('losers', doc_index=4)")
    print(f"  → {tfidf_score('losers', 4):.4f}")