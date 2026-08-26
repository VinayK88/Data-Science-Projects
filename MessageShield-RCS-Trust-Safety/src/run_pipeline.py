from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

from generate_data import generate_messages
from features import add_graph_features
from model import build_model, choose_threshold, evaluate
from experiments import ab_test_report, ipw_counterfactual
from monitoring import ecosystem_metrics, alert_on_shift


def main() -> None:
    parser = argparse.ArgumentParser(description="RCS/RBM Trust & Safety data science demo")
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = add_graph_features(generate_messages(args.rows))
    train, test = train_test_split(df, test_size=0.30, stratify=df.is_spam, random_state=42)
    model = build_model()
    model.fit(train, train.is_spam)
    p = model.predict_proba(test)[:, 1]
    threshold = choose_threshold(test.is_spam, p, max_fpr=0.02)
    ev = evaluate(test.is_spam, p, threshold)
    scored = test.copy()
    scored["spam_score"] = p
    scored.to_csv(outdir / "scored_messages.csv", index=False)
    ecosystem_metrics(scored, threshold).to_csv(outdir / "ecosystem_metrics.csv", index=False)
    joblib.dump(model, outdir / "spam_model.joblib")
    early = scored[scored.hour < 12].spam_score
    late = scored[scored.hour >= 12].spam_score
    summary = {"model_evaluation": ev.__dict__, "ab_test": ab_test_report(test),
               "counterfactual_ipw": ipw_counterfactual(test),
               "score_shift_monitor": alert_on_shift(early, late)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
