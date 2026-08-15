"""Offline evaluation and threshold optimization for RiskOS."""

from dataclasses import dataclass

from riskos.core import risk_score
from riskos.simulator import SyntheticCase, generate_cases


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    review_count: int
    missed_fraud_loss: float
    false_positive_cost: float
    review_cost: float
    capacity_penalty: float
    total_cost: float


def evaluate_threshold(
    cases: list[SyntheticCase],
    threshold: float,
    review_capacity: int = 60,
    loss_given_fraud: float = 0.25,
    false_positive_unit_cost: float = 400.0,
    analyst_review_unit_cost: float = 15.0,
    overflow_unit_cost: float = 250.0,
) -> ThresholdResult:
    """Evaluate one review threshold using model and operating metrics.

    `loss_given_fraud` converts gross exposure into an illustrative expected
    realized loss. The cost constants are scenario assumptions, not claims
    about any company's economics.
    """
    tp = fp = fn = 0
    missed_loss = 0.0
    review_count = 0

    for case in cases:
        score = risk_score(case.features)
        flagged = score >= threshold
        if flagged:
            review_count += 1
            if case.is_fraud:
                tp += 1
            else:
                fp += 1
        elif case.is_fraud:
            fn += 1
            missed_loss += case.features.exposure_usd * loss_given_fraud

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_cost = fp * false_positive_unit_cost
    review_cost = review_count * analyst_review_unit_cost
    overflow = max(0, review_count - review_capacity)
    capacity_penalty = overflow * overflow_unit_cost
    total_cost = missed_loss + false_positive_cost + review_cost + capacity_penalty

    return ThresholdResult(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        review_count=review_count,
        missed_fraud_loss=missed_loss,
        false_positive_cost=false_positive_cost,
        review_cost=review_cost,
        capacity_penalty=capacity_penalty,
        total_cost=total_cost,
    )


def threshold_sweep(cases: list[SyntheticCase], review_capacity: int = 60) -> list[ThresholdResult]:
    thresholds = [round(x / 100, 2) for x in range(15, 81, 5)]
    return [evaluate_threshold(cases, t, review_capacity=review_capacity) for t in thresholds]


def best_threshold(
    cases: list[SyntheticCase],
    review_capacity: int = 60,
    hard_capacity: bool = True,
) -> ThresholdResult:
    """Select the minimum-cost threshold, optionally requiring queue feasibility."""
    rows = threshold_sweep(cases, review_capacity)
    if hard_capacity:
        feasible = [row for row in rows if row.review_count <= review_capacity]
        if feasible:
            rows = feasible
    return min(rows, key=lambda row: row.total_cost)


def main() -> None:
    cases = generate_cases()
    rows = threshold_sweep(cases)
    best = best_threshold(cases)

    print("RiskOS threshold optimization")
    print("threshold precision recall   f1 reviews total_cost")
    for row in rows:
        marker = " *" if row == best else ""
        print(
            f"{row.threshold:8.2f} {row.precision:9.2%} {row.recall:6.2%} "
            f"{row.f1:5.2%} {row.review_count:7d} ${row.total_cost:10,.0f}{marker}"
        )
    print(
        f"\nSelected threshold={best.threshold:.2f}: minimum expected operating cost "
        "among thresholds that fit analyst review capacity."
    )


if __name__ == "__main__":
    main()
