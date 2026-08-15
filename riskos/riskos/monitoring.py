"""Lightweight drift and model-health utilities for RiskOS."""

from math import log


def _bucket_counts(values: list[float], bins: list[float]) -> list[int]:
    counts = [0] * (len(bins) - 1)
    for value in values:
        for i in range(len(bins) - 1):
            left, right = bins[i], bins[i + 1]
            if left <= value < right or (i == len(bins) - 2 and value == right):
                counts[i] += 1
                break
    return counts


def population_stability_index(
    reference: list[float],
    current: list[float],
    bins: list[float] | None = None,
) -> float:
    """Compute PSI with fixed score bins and small smoothing."""
    if not reference or not current:
        raise ValueError("reference and current distributions must be non-empty")
    bins = bins or [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
    ref_counts = _bucket_counts(reference, bins)
    cur_counts = _bucket_counts(current, bins)
    ref_total = sum(ref_counts)
    cur_total = sum(cur_counts)
    eps = 1e-6

    psi = 0.0
    for ref_count, cur_count in zip(ref_counts, cur_counts):
        ref_pct = max(ref_count / ref_total, eps)
        cur_pct = max(cur_count / cur_total, eps)
        psi += (cur_pct - ref_pct) * log(cur_pct / ref_pct)
    return psi


def drift_summary(reference: list[float], current: list[float]) -> dict[str, float | str]:
    psi = population_stability_index(reference, current)
    ref_mean = sum(reference) / len(reference)
    cur_mean = sum(current) / len(current)
    high_risk_rate = sum(score >= 0.78 for score in current) / len(current)

    if psi < 0.10:
        status = "stable"
    elif psi < 0.25:
        status = "watch"
    else:
        status = "investigate"

    return {
        "psi": psi,
        "reference_mean_score": ref_mean,
        "current_mean_score": cur_mean,
        "mean_score_shift": cur_mean - ref_mean,
        "current_high_risk_rate": high_risk_rate,
        "status": status,
    }
