# MessageShield: RCS Trust & Safety Data Science

A portfolio-grade Trust & Safety data science project for detecting **spam, phishing, and unwanted messaging traffic** in an RCS/RBM-style ecosystem while explicitly controlling harm to legitimate users.

The project combines **message text**, **sender behavior**, and **communication-graph signals**, then evaluates product interventions with **A/B testing** and **counterfactual inference**. It also produces ecosystem-health metrics for spam prevalence, false positives, enforcement efficacy, and user impact.

> This repository uses fully synthetic data. It is inspired by analytical problems common to large-scale messaging Trust & Safety systems and is not affiliated with or derived from Google production systems.

## Why this project

A production anti-abuse system cannot optimize only for model accuracy. Blocking legitimate conversations is costly, adversaries adapt, and product interventions can affect engagement. MessageShield therefore frames the problem around four questions:

1. **Detection:** Can we identify abusive messages using content, behavioral, and graph signals?
2. **User safety:** How much spam can we catch while holding the false-positive rate below a strict guardrail?
3. **Product impact:** Does a warning UI reduce risky click-through, and what side effects does it have?
4. **Ecosystem health:** Are spam prevalence, enforcement efficacy, or model scores shifting over time?

## Architecture

```text
Synthetic RCS/RBM events
        |
        v
Text signals + Behavioral signals + Communication graph
        |
        v
Spam classifier + operational threshold
        |
        +--------------------+
        |                    |
        v                    v
Ecosystem metrics      A/B test + IPW counterfactual
```

## What it demonstrates

- Python-based end-to-end analytics pipeline
- Statistical classification for spam/phishing detection
- NLP using TF-IDF word and bi-gram features
- Behavioral abuse features such as sending velocity and recipient fan-out
- Graph-derived sender features using NetworkX
- Imbalanced classification with PR-AUC and operational threshold selection
- False-positive guardrails to protect legitimate users
- A/B testing with a two-proportion z-test
- Counterfactual treatment-effect estimation using inverse propensity weighting
- Ecosystem-health monitoring and anomaly alerts
- Reproducible synthetic-data generation

## Modeling approach

The classifier combines text, behavioral and graph signals. The operational threshold is selected to maximize **spam recall while keeping false-positive rate ≤ 2%**, reflecting the product cost of blocking legitimate communication.

## Experimentation

`ab_test_report()` measures a safety-warning intervention with a two-proportion z-test. `ipw_counterfactual()` estimates the average treatment effect using inverse propensity weighting when exposure is observational rather than perfectly randomized.

## Core metrics

- Spam prevalence
- Enforcement rate
- False-positive rate
- Spam recall
- User click rate
- Model-score shift alert

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py --rows 30000
```

Outputs are written to `outputs/` as scored messages, ecosystem metrics, a serialized classifier and JSON experiment/model summaries.

## Example interview walkthrough

> I built an end-to-end Trust & Safety data science system for an RCS-style messaging ecosystem. I generated realistic synthetic message events, combined NLP features with sender behavior and communication-graph signals, and trained a spam classifier. Instead of optimizing only for accuracy, I selected the operating threshold to maximize abuse recall subject to a 2% false-positive guardrail, because blocking legitimate conversations is a key product risk. I then evaluated a warning UI with both an A/B test and inverse-propensity-weighted counterfactual analysis, and built monitoring for spam prevalence, enforcement rate, false positives, recall, user click-through, and score shifts. The project shows how I would connect modeling, experimentation, and product decision-making in a large messaging Trust & Safety environment.

## Repository structure

```text
MessageShield-RCS-Trust-Safety/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── model.py
│   ├── experiments.py
│   ├── monitoring.py
│   └── run_pipeline.py
└── tests/
    └── test_pipeline.py
```

## Production extensions

Possible next steps include transformer embeddings, graph embeddings, temporal concept drift, campaign simulation, explainability, uplift modeling, Streamlit monitoring, warehouse SQL features, and online/offline evaluation parity checks.
