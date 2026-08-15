from riskos.core import EntityFeatures, decision, expected_loss, reasons, risk_score


def sample_entities():
    return [
        EntityFeatures("carrier_001", 420, 0, 0, 1.1, 0, 0, 1200, 0),
        EntityFeatures("carrier_042", 5, 1, 1, 6.2, 4, 3, 20000, 1),
        EntityFeatures("carrier_108", 80, 1, 0, 2.4, 1, 0, 18000, 0),
        EntityFeatures("carrier_211", 14, 0, 1, 3.8, 2, 1, 65000, 1),
    ]


def main():
    ranked = []
    for entity in sample_entities():
        score = risk_score(entity)
        ranked.append((expected_loss(score, entity.exposure_usd), entity, score))

    ranked.sort(reverse=True, key=lambda row: row[0])

    print("RiskOS synthetic marketplace review queue")
    print("-" * 72)
    for loss, entity, score in ranked:
        print(
            f"entity={entity.entity_id:12s} "
            f"risk={score:.2f} "
            f"action={decision(score, entity.exposure_usd):9s} "
            f"expected_loss=${loss:,.0f}"
        )
        print("reasons=" + ", ".join(reasons(entity)))


if __name__ == "__main__":
    main()
