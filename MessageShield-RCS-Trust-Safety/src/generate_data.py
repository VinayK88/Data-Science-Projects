from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

SPAM_TERMS = ["urgent", "winner", "verify", "gift", "claim", "payment", "suspended", "crypto", "refund"]
HAM_TERMS = ["meeting", "thanks", "photo", "dinner", "tomorrow", "delivery", "family", "project", "hello"]
SHARED_TERMS = ["account", "message", "please", "link", "today", "service", "confirm", "update", "call", "order"]


def generate_messages(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sender_pool = np.array([f"sender_{i:05d}" for i in range(3500)])
    receiver_pool = np.array([f"user_{i:05d}" for i in range(12000)])
    senders = rng.choice(sender_pool, n)
    receivers = rng.choice(receiver_pool, n)
    sender_idx = np.array([int(s.split('_')[1]) for s in senders])
    risky_sender = (sender_idx < 260).astype(int)
    hour = rng.integers(0, 24, n)
    country_risk = rng.beta(1.4, 5.0, n)
    sender_age_days = np.maximum(1, rng.gamma(2.2, 180, n)).astype(int)
    messages_24h = rng.poisson(6 + 30 * risky_sender, n)
    unique_recipients_24h = np.maximum(1, rng.poisson(3 + 22 * risky_sender, n))
    url_count = rng.poisson(0.12 + 0.9 * risky_sender, n)
    phone_number_count = rng.binomial(2, 0.05 + 0.18 * risky_sender, n)
    uppercase_ratio = np.clip(rng.beta(1.3, 7.0, n) + 0.20 * risky_sender, 0, 1)
    repeated_text_score = np.clip(rng.beta(1.5, 5.0, n) + 0.35 * risky_sender, 0, 1)
    prior_reports = rng.poisson(0.05 + 1.7 * risky_sender, n)
    business_sender = rng.binomial(1, 0.18, n)
    logit = (-4.0 + 2.3 * risky_sender + 1.9 * repeated_text_score + 0.65 * url_count
             + 0.08 * unique_recipients_24h + 0.55 * prior_reports + 1.0 * country_risk
             + 0.6 * ((hour <= 5) | (hour >= 23)) - 0.0012 * sender_age_days - 0.45 * business_sender)
    prob = 1 / (1 + np.exp(-logit))
    is_spam = rng.binomial(1, np.clip(prob, 0.001, 0.98))
    texts = []
    for y in is_spam:
        k = int(rng.integers(5, 13))
        if y:
            primary, secondary, primary_prob = SPAM_TERMS, HAM_TERMS, 0.48
        else:
            primary, secondary, primary_prob = HAM_TERMS, SPAM_TERMS, 0.58
        words = []
        for _ in range(k):
            r = rng.random()
            if r < primary_prob:
                words.append(rng.choice(primary))
            elif r < 0.82:
                words.append(rng.choice(SHARED_TERMS))
            else:
                words.append(rng.choice(secondary))
        texts.append(" ".join(words))
    treatment_prob = np.clip(0.35 + 0.20 * country_risk + 0.15 * risky_sender, 0.05, 0.90)
    treatment = rng.binomial(1, treatment_prob)
    base_click_prob = np.clip(0.17 + 0.34 * is_spam + 0.05 * business_sender, 0.01, 0.95)
    click_prob = np.clip(base_click_prob - treatment * (0.20 * is_spam + 0.025 * (1-is_spam)), 0.005, 0.95)
    clicked = rng.binomial(1, click_prob)
    return pd.DataFrame({"message_id": [f"msg_{i:07d}" for i in range(n)], "sender_id": senders,
        "receiver_id": receivers, "text": texts, "hour": hour, "country_risk": country_risk,
        "sender_age_days": sender_age_days, "messages_24h": messages_24h,
        "unique_recipients_24h": unique_recipients_24h, "url_count": url_count,
        "phone_number_count": phone_number_count, "uppercase_ratio": uppercase_ratio,
        "repeated_text_score": repeated_text_score, "prior_reports": prior_reports,
        "business_sender": business_sender, "warning_ui": treatment, "clicked": clicked, "is_spam": is_spam})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/messages.csv")
    args = parser.parse_args()
    df = generate_messages(args.n, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}. Spam rate={df.is_spam.mean():.2%}")

if __name__ == "__main__":
    main()
