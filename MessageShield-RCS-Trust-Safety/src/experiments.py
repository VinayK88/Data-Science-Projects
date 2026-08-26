from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def ab_test_report(df: pd.DataFrame) -> dict:
    """Two-proportion z-test for warning UI vs control click-through."""
    a = df[df.warning_ui == 0].clicked
    b = df[df.warning_ui == 1].clicked
    p0, p1 = a.mean(), b.mean()
    n0, n1 = len(a), len(b)
    pooled = (a.sum() + b.sum()) / (n0 + n1)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    z = (p1 - p0) / max(se, 1e-12)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {"control_ctr": p0, "treatment_ctr": p1, "absolute_lift": p1 - p0,
            "relative_lift": (p1 - p0) / p0, "z": z, "p_value": p_value}


def ipw_counterfactual(df: pd.DataFrame) -> dict:
    """Estimate ATE for warning UI using inverse propensity weighting."""
    features = ["country_risk", "messages_24h", "unique_recipients_24h", "prior_reports", "business_sender"]
    X, t, y = df[features], df.warning_ui.values, df.clicked.values
    prop = LogisticRegression(max_iter=500).fit(X, t).predict_proba(X)[:, 1]
    prop = np.clip(prop, 0.03, 0.97)
    mu1 = np.sum(t * y / prop) / np.sum(t / prop)
    mu0 = np.sum((1 - t) * y / (1 - prop)) / np.sum((1 - t) / (1 - prop))
    return {"ipw_ctr_if_warning": mu1, "ipw_ctr_if_no_warning": mu0, "ate": mu1 - mu0}
