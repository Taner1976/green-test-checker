import numpy as np
import pandas as pd
from scipy import stats

def norm_choice(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ["", "NA", "NAN", "-", "."]:
        return np.nan
    return s

def parse_key(text, item_cols):
    """
    Accepts:
    - "B C A D ..."  OR "BCAD..."
    - "Item1=B, Item2=C, ..."  (item column names must match)
    """
    t = text.strip().upper()

    if "=" in t:
        mapping = {}
        parts = [p.strip() for p in t.replace(";", ",").split(",") if p.strip()]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                mapping[k.strip()] = v.strip()
        key = [mapping.get(col, np.nan) for col in item_cols]
        return pd.Series(key, index=item_cols).apply(norm_choice)

    t2 = t.replace(",", " ").replace("\n", " ").strip()
    toks = [tok for tok in t2.split() if tok]

    # single compact token "BCAD..."
    if len(toks) == 1 and len(toks[0]) == len(item_cols):
        toks = list(toks[0])

    if len(toks) != len(item_cols):
        return None

    return pd.Series(toks, index=item_cols).apply(norm_choice)

def kr20(X01: pd.DataFrame) -> float:
    k = X01.shape[1]
    if k < 2:
        return np.nan
    p = X01.mean(axis=0)
    q = 1 - p
    total = X01.sum(axis=1)
    var_total = total.var(ddof=1)
    if var_total == 0 or np.isnan(var_total):
        return np.nan
    return (k / (k - 1)) * (1 - (p * q).sum() / var_total)

def corrected_rpbis(X01: pd.DataFrame) -> pd.Series:
    """Corrected item-total correlation: corr(item, total-minus-item)."""
    total = X01.sum(axis=1)
    out = {}
    for col in X01.columns:
        item = X01[col]
        total_minus = total - item
        tmp = pd.concat([item, total_minus], axis=1).dropna()
        if tmp.shape[0] < 10 or tmp.iloc[:, 0].nunique() < 2:
            out[col] = np.nan
        else:
            out[col] = tmp.iloc[:, 0].corr(tmp.iloc[:, 1])
    return pd.Series(out)

def upper_lower_groups(total_scores: pd.Series, frac=0.27):
    n = len(total_scores)
    g = int(np.floor(n * frac))
    if g < 1:
        return None, None
    order = total_scores.sort_values(ascending=False)
    return order.index[:g], order.index[-g:]

def difficulty_category(p):
    if pd.isna(p): return ""
    if 0.00 <= p < 0.20: return "Very Difficult"
    if 0.20 <= p < 0.40: return "Difficult"
    if 0.40 <= p < 0.60: return "Average"
    if 0.60 <= p < 0.80: return "Easy"
    return "Very Easy"

def discrim_category(r):
    if pd.isna(r): return ""
    if r < 0.00: return "Discard"
    if 0.00 <= r < 0.20: return "Improve (Revision)"
    if 0.20 <= r < 0.30: return "Mediocre"
    if 0.30 <= r < 0.40: return "Good"
    return "Very Good"

def score_distribution_interpretation(total_scores, k_items):
    mean = float(total_scores.mean())
    sd = float(total_scores.std(ddof=1)) if len(total_scores) > 1 else 0.0
    skew = float(stats.skew(total_scores, bias=False)) if len(total_scores) > 2 else np.nan

    parts = [f"Mean={mean:.2f}, SD={sd:.2f}."]
    if not np.isnan(skew):
        if skew > 0.5:
            parts.append("Scores are **positively skewed** → test may be difficult for this group.")
        elif skew < -0.5:
            parts.append("Scores are **negatively skewed** → test may be easy for this group.")
        else:
            parts.append("Scores are **approximately symmetric** → difficulty seems balanced.")

    if sd < (0.15 * k_items):
        parts.append("Spread is **narrow**; items may be too easy/hard or not diverse enough.")
    else:
        parts.append("Spread looks **reasonable** for differentiation.")
    return " ".join(parts)

def unidimensionality_pca(X01: pd.DataFrame):
    """
    PCA on item correlation matrix (phi correlations for 0/1).
    Returns:
      eigvals (descending), message, var1 (approx variance explained by 1st)
    """
    C = X01.corr().fillna(0.0)
    eigvals = np.linalg.eigvalsh(C.to_numpy())
    eigvals = np.sort(eigvals)[::-1]

    if len(eigvals) == 0:
        return eigvals, "Not enough items for PCA.", np.nan

    first = eigvals[0]
    second = eigvals[1] if len(eigvals) > 1 else np.nan
    ratio = (first / second) if (len(eigvals) > 1 and second > 0) else np.nan
    var1 = first / len(eigvals)  # trace = k

    msg = (
        f"1st eigenvalue={first:.2f}, 2nd={second:.2f}, ratio(1st/2nd)={ratio:.2f}. "
        f"Approx. variance explained by 1st component={var1*100:.1f}%."
    )

    if len(eigvals) >= 10:
        if (not np.isnan(ratio) and ratio >= 3) and var1 >= 0.20:
            verdict = "Rule-of-thumb: **approximately unidimensional**."
        elif (not np.isnan(ratio) and ratio < 2) or var1 < 0.15:
            verdict = "Rule-of-thumb: **possible multidimensionality**; review blueprint/content."
        else:
            verdict = "Rule-of-thumb: **borderline**; interpret cautiously."
    else:
        verdict = "Few items: PCA may be unstable; interpret cautiously."

    return eigvals, msg + " " + verdict, var1

def detect_options(resp_df: pd.DataFrame, allowed=("A","B","C","D","E")):
    """
    Detect which options exist in the response data (A/B/C/D/E).
    """
    vals = resp_df.stack().dropna().astype(str).str.upper().unique().tolist()
    opts = [o for o in allowed if o in vals]
    return opts if opts else list(allowed)
