"""
Toxicity Bias Audit Pipeline
Uses identity columns directly from dataset to stratify and generate counterfactuals.

Usage:
    python bias_audit.py /path/to/dataset.csv [--top N] [--middle N] [--bottom N]

Example:
    python bias_audit.py /mnt/user-data/uploads/data_first_1k.csv --top 100 --middle 100 --bottom 100
"""

import csv
import re
import json
import argparse
from pathlib import Path
from typing import Generator, Dict, Any, List, Tuple
from dataclasses import dataclass


# Identity columns in Jigsaw dataset -> actual terms to swap in text (include singular + plural)
IDENTITY_MAPPING = {
    'male': ['man', 'men', 'guy', 'guys', 'male', 'males', 'boy', 'boys'],
    'female': ['woman', 'women', 'girl', 'girls', 'female', 'females', 'lady', 'ladies'],
    'transgender': ['transgender', 'trans'],
    'heterosexual': ['straight'],
    'homosexual_gay_or_lesbian': ['gay', 'gays', 'lesbian', 'lesbians', 'queer'],
    'bisexual': ['bisexual', 'bisexuals'],
    'christian': ['christian', 'christians'],
    'jewish': ['jewish', 'jew', 'jews'],
    'muslim': ['muslim', 'muslims', 'islamic'],
    'hindu': ['hindu', 'hindus'],
    'buddhist': ['buddhist', 'buddhists'],
    'atheist': ['atheist', 'atheists'],
    'black': ['black', 'blacks', 'african', 'africans'],
    'white': ['white', 'whites', 'caucasian', 'caucasians'],
    'asian': ['asian', 'asians'],
    'latino': ['latino', 'latinos', 'latina', 'latinas', 'hispanic', 'hispanics'],
    'physical_disability': [],
    'intellectual_or_learning_disability': [],
    'psychiatric_or_mental_illness': [],
    'other_disability': [],
}

# Swap pairs: if a term from one group appears, swap to term from another group in same category
CATEGORY_GROUPS = {
    'gender': ['male', 'female', 'transgender'],
    'sexuality': ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual'],
    'religion': ['christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist'],
    'race': ['black', 'white', 'asian', 'latino'],
}


def row_generator(filename: str) -> Generator[Dict[str, Any], None, None]:
    """Stream rows from CSV without loading entire file into memory."""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def get_identity_mentions(row: Dict[str, Any], threshold: float = 0.5) -> List[str]:
    """
    Get list of identity columns that are mentioned in this row.
    
    Args:
        row: CSV row as dict
        threshold: Minimum value (0-1) to count as mention
        
    Returns:
        List of identity column names that meet threshold
    """
    mentions = []
    for col in IDENTITY_MAPPING.keys():
        try:
            val = float(row.get(col, 0) or 0)
            if val >= threshold:
                mentions.append(col)
        except (ValueError, TypeError):
            pass
    return mentions


def filter_and_sort(filename: str, threshold: float = 0.5) -> List[Tuple[Dict[str, Any], float]]:
    """
    Stream through file, filter for identity mentions, and return sorted list.
    
    Args:
        filename: Path to CSV
        threshold: Minimum identity mention threshold
        
    Returns:
        List of (row, toxicity_score) tuples sorted by toxicity
    """
    print(f"Scanning {filename}...")
    filtered = []
    count = 0
    
    for row in row_generator(filename):
        mentions = get_identity_mentions(row, threshold)
        
        if mentions:
            try:
                toxicity = float(row.get('toxicity', 0) or 0)
                filtered.append((row, toxicity))
                count += 1
                if count % 50000 == 0:
                    print(f"  Found {count} rows with identity mentions...")
            except (ValueError, TypeError):
                pass
    
    print(f"Total rows with identity mentions: {count}")
    
    # Sort by toxicity
    filtered.sort(key=lambda x: x[1])
    return filtered


def stratify(filtered_rows: List[Tuple[Dict[str, Any], float]], 
             top_n: int = 100, 
             middle_n: int = 100, 
             bottom_n: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split rows into three strata by toxicity.
    
    Args:
        filtered_rows: Sorted list of (row, toxicity) tuples
        top_n: Number of highest toxicity samples
        middle_n: Number of middle toxicity samples
        bottom_n: Number of lowest toxicity samples
        
    Returns:
        Dict with 'top', 'middle', 'bottom' keys mapping to lists of rows
    """
    result = {
        'bottom': [row for row, _ in filtered_rows[:bottom_n]],
        'top': [row for row, _ in filtered_rows[-top_n:]] if len(filtered_rows) > 0 else [],
    }
    
    if len(filtered_rows) > 0:
        median_idx = len(filtered_rows) // 2
        start_idx = max(0, median_idx - middle_n // 2)
        end_idx = min(len(filtered_rows), start_idx + middle_n)
        result['middle'] = [row for row, _ in filtered_rows[start_idx:end_idx]]
    else:
        result['middle'] = []
    
    # Print summary
    if len(result['bottom']) > 0:
        print(f"Bottom {len(result['bottom'])}: toxicity {float(filtered_rows[0][1]):.4f} - {float(filtered_rows[len(result['bottom'])-1][1]):.4f}")
    if len(result['middle']) > 0:
        print(f"Middle {len(result['middle'])}: toxicity {float(filtered_rows[start_idx][1]):.4f} - {float(filtered_rows[end_idx-1][1]):.4f}")
    if len(result['top']) > 0:
        print(f"Top {len(result['top'])}: toxicity {float(filtered_rows[-len(result['top'])][1]):.4f} - {float(filtered_rows[-1][1]):.4f}")
    
    return result


def get_swap_target(mentioned_col: str) -> str:
    """
    For a mentioned identity column, pick another one from same category to swap to.
    
    Args:
        mentioned_col: Identity column name (e.g., 'black')
        
    Returns:
        Another column name in same category, or original if no swap available
    """
    # Find which category this column belongs to
    for category, cols in CATEGORY_GROUPS.items():
        if mentioned_col in cols:
            # Pick the first other column in this category
            others = [c for c in cols if c != mentioned_col]
            if others:
                return others[0]
    return mentioned_col


def generate_counterfactual(comment: str, mentioned_col: str) -> str:
    """
    Generate a counterfactual by swapping ALL identity terms in the comment.
    
    Finds which category mentioned_col belongs to, then swaps ALL terms from that category.
    E.g., if 'female' is mentioned, swaps 'woman', 'girl', 'female', 'lady' → corresponding male terms.
    
    Args:
        comment: Original comment text
        mentioned_col: Identity column that's mentioned (e.g., 'female')
        
    Returns:
        Modified comment with all identity terms in that category swapped
    """
    # Find which category this column belongs to
    category = None
    for cat, cols in CATEGORY_GROUPS.items():
        if mentioned_col in cols:
            category = cat
            break
    
    if not category:
        return comment
    
    # Get all columns in this category
    category_cols = CATEGORY_GROUPS[category]
    
    # Find all the terms from all columns in this category
    all_original_terms = []
    for col in category_cols:
        all_original_terms.extend(IDENTITY_MAPPING.get(col, []))
    
    if not all_original_terms:
        return comment
    
    # Get swap target and its terms
    swap_col = get_swap_target(mentioned_col)
    if swap_col == mentioned_col:
        return comment
    
    swap_terms = IDENTITY_MAPPING.get(swap_col, [])
    if not swap_terms:
        return comment
    
    # Build mapping: each original term → first swap term
    # (This ensures all gender terms swap consistently)
    term_map = {}
    for orig_term in all_original_terms:
        term_map[orig_term.lower()] = swap_terms[0]
    
    # Create regex pattern for ANY term in any column of this category
    pattern = re.compile(r'\b(' + '|'.join(re.escape(t) for t in all_original_terms if t) + r')\b', re.IGNORECASE)
    
    # Replace each match with swap term, preserving case
    def replace_func(match):
        original = match.group(0)
        lookup_key = original.lower()
        swap = term_map.get(lookup_key, original)
        
        # Try to match case
        if original and original[0].isupper():
            return swap[0].upper() + swap[1:] if len(swap) > 1 else swap.upper()
        return swap
    
    swapped = pattern.sub(replace_func, comment)
    return swapped if swapped != comment else comment


def save_counterfactuals(rows: List[Dict[str, Any]], output_file: str) -> None:
    """
    Generate counterfactuals from rows and save to CSV.
    
    For each row with identity mentions, creates:
    - One row with original comment
    - One row for each mentioned identity with a counterfactual swap
    
    Args:
        rows: List of row dicts
        output_file: Where to save
    """
    output_rows = []
    
    for i, row in enumerate(rows):
        comment = row.get('comment_text', '')
        row_id = row.get('id', '')
        
        if not comment:
            continue
        
        mentions = get_identity_mentions(row)
        if not mentions:
            continue
        
        # Add original
        out_row = {
            'id': row_id,
            'variant_id': 'original',
            'variant_text': comment,
            'is_original': 'true',
            'original_toxicity': row.get('toxicity', ''),
        }
        output_rows.append(out_row)
        
        # Add counterfactuals (one per mentioned identity)
        for j, mentioned_col in enumerate(mentions):
            counterfactual = generate_counterfactual(comment, mentioned_col)
            
            if counterfactual != comment:  # Only save if different
                out_row = {
                    'id': row_id,
                    'variant_id': f'swap_{mentioned_col}',
                    'variant_text': counterfactual,
                    'is_original': 'false',
                    'original_toxicity': row.get('toxicity', ''),
                }
                output_rows.append(out_row)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1} rows, generated {len(output_rows)} variants...")
    
    # Save to CSV
    if output_rows:
        fieldnames = ['id', 'variant_id', 'variant_text', 'is_original', 'original_toxicity']
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        
        print(f"Saved {len(output_rows)} variants to {output_file}")


def run_audit(csv_file: str, output_dir: str = 'output',
              top_n: int = 100, middle_n: int = 100, bottom_n: int = 100) -> None:
    """
    Run full audit pipeline on a CSV file.
    
    Args:
        csv_file: Path to input CSV
        output_dir: Where to save results
        top_n: Number of top toxicity samples
        middle_n: Number of middle toxicity samples
        bottom_n: Number of bottom toxicity samples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("TOXICITY BIAS AUDIT")
    print("="*70)
    
    # Step 1: Filter and sort
    print("\n[1/3] Filtering and sorting by toxicity...")
    filtered_rows = filter_and_sort(csv_file)
    
    if not filtered_rows:
        print("No rows with identity mentions found!")
        return
    
    # Step 2: Stratify
    print(f"\n[2/3] Stratifying into {top_n}/{middle_n}/{bottom_n}...")
    strata = stratify(filtered_rows, top_n=top_n, middle_n=middle_n, bottom_n=bottom_n)
    
    # Step 3: Generate counterfactuals for each stratum
    print("\n[3/3] Generating counterfactuals...")
    
    results = {}
    for stratum_name in ['top', 'middle', 'bottom']:
        rows = strata[stratum_name]
        if not rows:
            print(f"  Skipping {stratum_name} (no rows)")
            continue
        
        output_file = output_dir / f"counterfactuals_{stratum_name}.csv"
        print(f"\n  {stratum_name.upper()} stratum ({len(rows)} rows):")
        save_counterfactuals(rows, str(output_file))
        results[stratum_name] = str(output_file)
    
    print("\n" + "="*70)
    print("AUDIT COMPLETE")
    print("="*70)
    print(f"\nCounterfactual CSVs ready for API querying:")
    for stratum, filepath in results.items():
        print(f"  {stratum}: {filepath}")
    
    print(f"\nNext steps:")
    print(f"  1. Query Perspective API with each counterfactuals_*.csv file")
    print(f"  2. Collect scores in output CSV")
    print(f"  3. Analyze score differences within each comment variant group")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toxicity bias audit with counterfactual generation")
    parser.add_argument("csv_file", help="Path to CSV file")
    parser.add_argument("--top", type=int, default=100, help="Number of top toxicity samples")
    parser.add_argument("--middle", type=int, default=100, help="Number of middle toxicity samples")
    parser.add_argument("--bottom", type=int, default=100, help="Number of bottom toxicity samples")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    run_audit(args.csv_file, output_dir=args.output, top_n=args.top, middle_n=args.middle, bottom_n=args.bottom)