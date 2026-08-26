from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC = ["hour", "country_risk", "sender_age_days", "messages_24h", "unique_recipients_24h", "url_count",
           "phone_number_count", "uppercase_ratio", "repeated_text_score", "prior_reports", "business_sender",
           "sender_out_degree", "sender_weighted_out", "sender_pagerank", "fanout_ratio"]

@dataclass
class Evaluation:
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    spam_prevalence: float
    predicted_spam_rate: float


def build_model() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    pre = ColumnTransformer([("text", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=12000), "text"),
                             ("num", numeric, NUMERIC)])
    return Pipeline([("features", pre), ("model", LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear"))])


def choose_threshold(y_true: pd.Series, p: np.ndarray, max_fpr: float = 0.02) -> float:
    best_t, best_recall = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)
        if fpr <= max_fpr and recall > best_recall:
            best_t, best_recall = float(t), recall
    return best_t


def evaluate(y_true: pd.Series, p: np.ndarray, threshold: float) -> Evaluation:
    pred = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return Evaluation(threshold, roc_auc_score(y_true, p), average_precision_score(y_true, p), precision, recall, f1,
                      fp / max(fp + tn, 1), float(np.mean(y_true)), float(np.mean(pred)))
