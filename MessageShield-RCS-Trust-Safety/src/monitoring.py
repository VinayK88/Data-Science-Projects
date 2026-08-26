from __future__ import annotations
import numpy as np
import pandas as pd


def ecosystem_metrics(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    x = scored.copy()
    x["pred_spam"] = (x.spam_score >= threshold).astype(int)
    x["bucket"] = pd.cut(x.hour, bins=[-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])
    rows = []
    for bucket, g in x.groupby("bucket", observed=True):
        legit = g[g.is_spam == 0]
        spam = g[g.is_spam == 1]
        rows.append({"time_bucket": str(bucket), "messages": len(g), "spam_prevalence": g.is_spam.mean(),
                     "enforcement_rate": g.pred_spam.mean(),
                     "false_positive_rate": legit.pred_spam.mean() if len(legit) else np.nan,
                     "spam_recall": spam.pred_spam.mean() if len(spam) else np.nan,
                     "user_click_rate": g.clicked.mean()})
    return pd.DataFrame(rows)


def alert_on_shift(reference: pd.Series, current: pd.Series, z: float = 3.0) -> dict:
    ref_mean = reference.mean()
    ref_std = reference.std(ddof=1)
    cur_mean = current.mean()
    score = (cur_mean - ref_mean) / max(ref_std, 1e-12)
    return {"reference_mean": ref_mean, "current_mean": cur_mean, "z_score": score, "alert": bool(abs(score) >= z)}
