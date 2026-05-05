# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (toxicity-bias)
#     language: python
#     name: toxicity-bias
# ---

# %% [markdown]
# # Fairness Audit
#
# Miniature starting point for the Jigsaw unintended bias project.
#
# Goal: compare toxicity scores for counterfactual sentences where only one identity term changes.

# %% [markdown]
# ## 1. Setup

# %%
from pathlib import Path

import pandas as pd

DATA_PATH = Path("../data/train.csv")
SAMPLE_SIZE = 5000

# %% [markdown]
# ## 2. Load a small sample

# %%
df = pd.read_csv(DATA_PATH, usecols=["comment_text", "target"], nrows=SAMPLE_SIZE)
df.head()

# %%
df["target"].describe()

# %% [markdown]
# ## 3. Identity terms for counterfactual testing

# %%
identity_terms = {
    "religion": ["christian", "muslim", "jewish"],
    "gender": ["man", "woman"],
    "sexuality": ["straight", "gay"],
    "race": ["white", "black", "asian"],
}

identity_terms


# %% [markdown]
# ## 4. Create counterfactual sentences

# %%
def make_counterfactuals(template: str, terms: list[str]) -> pd.DataFrame:
    """Fill a sentence template with identity terms."""
    rows = []
    for term in terms:
        rows.append({
            "identity_term": term,
            "sentence": template.format(identity=term),
        })
    return pd.DataFrame(rows)


template = "I had a conversation with a {identity} person yesterday."
counterfactuals = make_counterfactuals(template, identity_terms["religion"])
counterfactuals


# %% [markdown]
# ## 5. Placeholder: train our TF-IDF Logistic Regression model
#
# Later we can add:
#
# - `TfidfVectorizer`
# - `LogisticRegression`
# - train/test split
# - toxicity score predictions for the counterfactual sentences

# %%
# TODO: Train baseline model here.
# Suggested next packages: scikit-learn, lime.

def score_with_local_model(sentences: list[str]) -> list[float]:
    raise NotImplementedError("Train the TF-IDF Logistic Regression model first.")


# %% [markdown]
# ## 6. Placeholder: score with Perspective API
#
# Later we can call the API here and store scores in a separate column.

# %%
# TODO: Add Perspective API scoring here.

def score_with_perspective_api(sentences: list[str]) -> list[float]:
    raise NotImplementedError("Add Perspective API key and request code first.")


# %% [markdown]
# ## 7. Placeholder: compare score differences
#
# The core audit table should eventually look like this: one row per counterfactual sentence, with scores from each model.

# %%
audit_results = counterfactuals.copy()
audit_results["local_model_score"] = pd.NA
audit_results["perspective_score"] = pd.NA

audit_results
