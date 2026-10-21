"""
Statistical Treatment section of the SOP, implemented directly:

RQ1 (English vs. Taglish, 20 matched pairs) and RQ3 (proposed framework vs.
REVIE, N matched pairs) both follow the same recipe per metric:
  1. Shapiro-Wilk on the per-pair DIFFERENCE scores decides normal vs. not.
  2. Normal  -> paired-samples t-test,        effect size = Cohen's d (d_z).
     Not     -> Wilcoxon signed-rank test,    effect size = rank-biserial r.
  3. alpha = 0.05. Every significant result is reported with its effect size,
     since a p-value alone doesn't say whether a difference is practically
     meaningful (SOP, "Effect Size" section).

RQ2 (trigger/language detection) is reported descriptively only — Trigger
Precision, Trigger Recall (must be >= 0.90), False Positive Rate (must be
< 0.10) — no significance test, per the SOP's own Table 3.

Nothing here calls an LLM; it only needs the numeric/label lists the
evaluation runner already produced.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

ALPHA = 0.05

# Fewer non-tied pairs than this and the rank-biserial r is reported but NOT
# labeled. With 2 non-zero differences r can only be -1, 0 or +1, so a
# "large effect" label there (RQ1 Groundedness in the first run: r = 1.0,
# p = 0.5, 16 of 18 pairs tied) describes the arithmetic, not the systems.
MIN_NONZERO_FOR_EFFECT_LABEL = 6


# Effect-size bands, per the SOP's "Effect Size" section (p. 69):
#   Cohen's d:          < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large
#   rank-biserial r:    "r values above 0.5 indicate a large practical effect" —
#                       i.e. Cohen's (1988) conventions for correlation-type
#                       effects: < 0.1 negligible, 0.1-0.3 small, 0.3-0.5
#                       medium, >= 0.5 large. (Using the d cut-offs for r would
#                       under-label it: r = 0.72 is large, not medium.)
_EFFECT_BANDS = {
    "cohens_d": (0.2, 0.5, 0.8),
    "rank_biserial_r": (0.1, 0.3, 0.5),
}


def _effect_label(magnitude: Optional[float], kind: str = "cohens_d") -> Optional[str]:
    if magnitude is None:
        return None
    small, medium, large = _EFFECT_BANDS[kind]
    m = abs(magnitude)
    if m < small:
        return "negligible"
    if m < medium:
        return "small"
    if m < large:
        return "medium"
    return "large"


@dataclass
class PairedComparison:
    metric_name: str
    n_pairs: int                      # pairs actually compared (both sides defined)
    n_dropped: int                    # pairs dropped for a missing/None score on either side
    mean_a: Optional[float]
    mean_b: Optional[float]
    mean_diff: Optional[float]        # mean(a - b)
    normality_p: Optional[float] = None
    is_normal: Optional[bool] = None
    test_used: Optional[str] = None   # "paired t-test" | "wilcoxon signed-rank" | None
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    significant: Optional[bool] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None   # "cohens_d" | "rank_biserial_r"
    effect_size_label: Optional[str] = None
    n_ties_dropped: int = 0           # zero-difference pairs Wilcoxon excluded (if that test ran)
    note: Optional[str] = None        # why a test couldn't run, or another caveat


def _paired_arrays(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Keeps only indices where BOTH a[i] and b[i] are real numbers (not
    None). A judge failure or a metric that's structurally undefined for an
    item (see ragas_metrics.py — e.g. Groundedness on a refusal with zero
    claims) must not silently become 0 or be imputed; it's dropped from THIS
    comparison and the drop count is reported, since a lot of drops is itself
    worth knowing (e.g. one system refuses far more than the other)."""
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be the same length ({len(a)} vs {len(b)})")
    kept_a, kept_b = [], []
    for x, y in zip(a, b):
        if x is not None and y is not None:
            kept_a.append(float(x))
            kept_b.append(float(y))
    return np.array(kept_a), np.array(kept_b), len(a) - len(kept_a)


def paired_comparison(a: Sequence[Optional[float]], b: Sequence[Optional[float]], metric_name: str = "", alpha: float = ALPHA) -> PairedComparison:
    """
    Compare two matched-pairs samples (e.g. English scores vs. Taglish
    scores for RQ1, or Sagot AI vs. REVIE scores for RQ3) for one metric.
    `a` and `b` must be same-length, index-aligned lists of scores (or None
    for an undefined score); the difference is always a[i] - b[i], so a
    positive mean_diff means `a` scored higher.
    """
    arr_a, arr_b, n_dropped = _paired_arrays(a, b)
    n = len(arr_a)
    result = PairedComparison(
        metric_name=metric_name, n_pairs=n, n_dropped=n_dropped,
        mean_a=float(arr_a.mean()) if n else None,
        mean_b=float(arr_b.mean()) if n else None,
        mean_diff=float((arr_a - arr_b).mean()) if n else None,
    )
    if n < 3:
        result.note = f"only {n} usable pair(s) after dropping missing scores — too few for any test (need >= 3)"
        return result

    diffs = arr_a - arr_b
    if np.allclose(diffs, diffs[0]):
        result.note = "every pair has the identical difference — no variance to test"
        result.test_used, result.statistic, result.p_value = "none (zero variance)", None, None
        return result

    shapiro_stat, shapiro_p = scipy_stats.shapiro(diffs)
    result.normality_p = float(shapiro_p)
    # bool(): comparing scipy's numpy.float64 gives numpy.bool, which
    # json.dump (summary.json) and FastAPI both refuse to serialize.
    result.is_normal = bool(shapiro_p > alpha)

    if result.is_normal:
        t_stat, p_value = scipy_stats.ttest_rel(arr_a, arr_b)
        std_diff = diffs.std(ddof=1)
        cohens_d = float(diffs.mean() / std_diff) if std_diff > 0 else None
        result.test_used = "paired t-test"
        result.statistic, result.p_value = float(t_stat), float(p_value)
        result.effect_size, result.effect_size_type = cohens_d, "cohens_d"
        result.effect_size_label = _effect_label(cohens_d, "cohens_d")
    else:
        nonzero = diffs[diffs != 0]
        result.n_ties_dropped = n - len(nonzero)
        if len(nonzero) < 1:
            result.note = "every pair is exactly tied (zero difference) — Wilcoxon cannot run"
            return result
        w_stat, p_value = scipy_stats.wilcoxon(nonzero, zero_method="wilcox", mode="auto")
        ranks = scipy_stats.rankdata(np.abs(nonzero))
        r_plus = ranks[nonzero > 0].sum()
        r_minus = ranks[nonzero < 0].sum()
        rank_biserial = float((r_plus - r_minus) / (r_plus + r_minus)) if (r_plus + r_minus) else None
        result.test_used = "wilcoxon signed-rank"
        result.statistic, result.p_value = float(w_stat), float(p_value)
        result.effect_size, result.effect_size_type = rank_biserial, "rank_biserial_r"
        if len(nonzero) >= MIN_NONZERO_FOR_EFFECT_LABEL:
            result.effect_size_label = _effect_label(rank_biserial, "rank_biserial_r")
        else:
            result.effect_size_label = None
            result.note = (
                f"only {len(nonzero)} non-tied pair(s) ({result.n_ties_dropped} tied) — the "
                f"rank-biserial r is not interpretable as an effect size at this n; report "
                f"the tie count instead of an effect-size label"
            )

    result.significant = result.p_value is not None and result.p_value < alpha
    return result


# ── RQ2: trigger/language-detection confusion matrix ──────────────────────

MIN_TRIGGER_RECALL = 0.90   # SOP: "no more than 10% of Taglish queries may be missed"
MAX_FALSE_POSITIVE_RATE = 0.10   # SOP: "an FPR below 0.10 acceptable"


@dataclass
class TriggerMetrics:
    tp: int
    fn: int
    fp: int
    tn: int
    n: int
    precision: Optional[float]
    recall: Optional[float]
    fpr: Optional[float]
    meets_recall_threshold: Optional[bool]   # recall >= MIN_TRIGGER_RECALL
    meets_fpr_threshold: Optional[bool]      # fpr < MAX_FALSE_POSITIVE_RATE


def _safe_div(numerator: int, denominator: int) -> Optional[float]:
    """None (not 0.0) when the denominator is 0 — e.g. FPR is undefined, not
    zero, if the test set contained no English queries at all; silently
    reporting 0.0 there would misleadingly look like a perfect result."""
    return numerator / denominator if denominator else None


def confusion_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> TriggerMetrics:
    """
    `y_true` / `y_pred` are 0/1 labels: 1 = Taglish, 0 = Non-Taglish (English),
    matching the SOP's RQ2 convention exactly (TP = Taglish correctly
    flagged, FN = Taglish missed, FP = English wrongly flagged as Taglish,
    TN = English correctly left alone).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be the same length ({len(y_true)} vs {len(y_pred)})")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    return TriggerMetrics(
        tp=tp, fn=fn, fp=fp, tn=tn, n=len(y_true),
        precision=_safe_div(tp, tp + fp),
        recall=recall,
        fpr=fpr,
        meets_recall_threshold=(recall >= MIN_TRIGGER_RECALL) if recall is not None else None,
        meets_fpr_threshold=(fpr < MAX_FALSE_POSITIVE_RATE) if fpr is not None else None,
    )