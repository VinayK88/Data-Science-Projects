"""Deterministic synthetic marketplace generator for RiskOS."""

from dataclasses import dataclass
from random import Random

from riskos.core import EntityFeatures


@dataclass(frozen=True)
class SyntheticCase:
    features: EntityFeatures
    is_fraud: int
    ring_id: str | None


def generate_cases(n: int = 600, fraud_rate: float = 0.14, seed: int = 17) -> list[SyntheticCase]:
    """Generate overlapping benign and fraud populations.

    Some fraud is deliberately stealthy and some legitimate activity contains
    risky-looking signals. That overlap makes threshold selection, false
    positives, review capacity, and operating cost visible in the evaluation.
    Labels remain fully synthetic and are not estimates of production behavior.
    """
    rng = Random(seed)
    cases: list[SyntheticCase] = []

    for idx in range(n):
        fraud = int(rng.random() < fraud_rate)
        ring_id = f"ring_{rng.randint(1, 8):02d}" if fraud and rng.random() < 0.58 else None

        if fraud:
            stealth = rng.random() < 0.35
            account_age = rng.randint(20, 700) if stealth else rng.randint(1, 180)
            new_device = int(rng.random() < (0.35 if stealth else 0.65))
            bank_change = int(rng.random() < (0.25 if stealth else 0.50))
            velocity = round(
                rng.uniform(1.1, 3.2) if stealth else rng.uniform(1.8, 7.0), 2
            )
            shared_devices = rng.randint(1, 4) if ring_id else rng.randint(0, 2)
            suspended_neighbors = (
                rng.randint(0, 3) if ring_id else int(rng.random() < 0.15)
            )
            exposure = round(rng.uniform(2500, 85000), 2)
            suspicious_sequence = int(rng.random() < (0.32 if stealth else 0.60))
        else:
            account_age = rng.randint(20, 1200)
            new_device = int(rng.random() < 0.22)
            bank_change = int(rng.random() < 0.11)
            velocity = (
                round(rng.uniform(2.0, 6.0), 2)
                if rng.random() < 0.12
                else round(max(0.3, rng.gauss(1.15, 0.50)), 2)
            )
            shared_devices = rng.randint(1, 3) if rng.random() < 0.20 else 0
            suspended_neighbors = int(rng.random() < 0.055)
            exposure = round(rng.uniform(100, 55000), 2)
            suspicious_sequence = int(rng.random() < 0.08)

        features = EntityFeatures(
            entity_id=f"carrier_{idx:04d}",
            account_age_days=account_age,
            new_device=new_device,
            bank_change_24h=bank_change,
            velocity_ratio=velocity,
            shared_device_count=shared_devices,
            suspended_neighbor_count=suspended_neighbors,
            exposure_usd=exposure,
            suspicious_sequence=suspicious_sequence,
        )
        cases.append(SyntheticCase(features, fraud, ring_id))

    return cases
