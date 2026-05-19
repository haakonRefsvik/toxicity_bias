"""
Identity bias exploration — Jigsaw Unintended Bias dataset
Run: python code/explore_identity_bias.py
Outputs a summary of toxicity distributions per identity group.
"""

import pandas as pd
import numpy as np

DATA_PATH = "data/train.csv"

IDENTITY_COLS = [
    "muslim", "black", "white", "jewish", "christian", "atheist",
    "asian", "buddhist", "hindu", "latino",
    "female", "male", "transgender",
    "homosexual_gay_or_lesbian", "heterosexual", "bisexual",
]

print("Loading data...")
df = pd.read_csv(DATA_PATH, usecols=["target", "comment_text"] + IDENTITY_COLS)
print(f"Total rows: {len(df):,}\n")

# ── 1. Overall toxicity distribution ─────────────────────────────────────────
print("=" * 55)
print("OVERALL TOXICITY DISTRIBUTION")
print("=" * 55)
bins = [0, 0.2, 0.5, 0.8, 1.0]
labels = ["low (0–0.2)", "mid (0.2–0.5)", "high (0.5–0.8)", "very high (0.8–1.0)"]
df["toxicity_band"] = pd.cut(df["target"], bins=bins, labels=labels)
print(df["toxicity_band"].value_counts().sort_index().to_string())
print()

# ── 2. Mean toxicity per identity group ──────────────────────────────────────
print("=" * 55)
print("MEAN TOXICITY SCORE PER IDENTITY GROUP")
print("(only comments where the identity column > 0.5)")
print("=" * 55)

results = []
for col in IDENTITY_COLS:
    if col not in df.columns:
        continue
    subset = df[df[col] >= 0.5]
    if len(subset) < 50:
        continue
    results.append({
        "identity": col,
        "n_comments": len(subset),
        "mean_toxicity": subset["target"].mean(),
        "pct_toxic": (subset["target"] >= 0.5).mean() * 100,
    })

results_df = pd.DataFrame(results).sort_values("mean_toxicity", ascending=False)
print(results_df.to_string(index=False))
print()

# ── 3. Sample comments at each toxicity level for each identity ───────────────
print("=" * 55)
print("SAMPLE COMMENTS PER IDENTITY × TOXICITY LEVEL")
print("=" * 55)

SAMPLE_IDENTITIES = ["muslim", "black", "white", "christian", "female", "homosexual_gay_or_lesbian"]

for identity in SAMPLE_IDENTITIES:
    if identity not in df.columns:
        continue
    subset = df[df[identity] >= 0.5].dropna(subset=["target", "comment_text"])
    if len(subset) == 0:
        continue

    print(f"\n--- {identity.upper()} ---")
    for band, lo, hi in [("LOW", 0.0, 0.2), ("MID", 0.3, 0.6), ("HIGH", 0.7, 1.0)]:
        sample = subset[(subset["target"] >= lo) & (subset["target"] < hi)]
        if len(sample) == 0:
            continue
        row = sample.sample(1, random_state=42).iloc[0]
        text = row["comment_text"][:200].replace("\n", " ")
        print(f"  [{band} | score={row['target']:.2f}] {text}")
