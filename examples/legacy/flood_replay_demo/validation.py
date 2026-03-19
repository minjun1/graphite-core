#!/usr/bin/env python3
"""
Hurricane Harvey (2017) — Observed Outcome Validation

Compares model-predicted blast radius ranking against actual observed
shutdown/impact data. Tests whether real AlphaEarth embeddings improve
the rank correlation.

Sources:
  - EIA Hurricane Harvey situation reports (eia.gov)
  - NWS Corpus Christi post-tropical cyclone report (weather.gov)
  - CSB Arkema Crosby investigation (csb.gov)
  - Port Houston Authority operations (porthouston.com)
  - DOE infrastructure situation reports (energy.gov)
  - Reuters, Forbes, Washington Post industry coverage
"""

import json
import os
from typing import Dict, List, Optional, Tuple


# ── Observed Outcomes ──
# Each facility's actual Harvey impact, from public reports.
# severity_rank: 1 = most severely impacted, higher = less impacted
# shutdown_days: approximate total days offline or severely impaired

OBSERVED_OUTCOMES = {
    "facility:MOTIVA_PORT_ARTHUR": {
        "facility": "Motiva Port Arthur Refinery",
        "impact": "Full shutdown. Largest US refinery (607k bbl/d). Controlled shutdown Aug 30, restart began Sep 11.",
        "shutdown_days": 12,
        "severity": "EXTREME",
        "severity_rank": 1,
        "source": "EIA situation reports; Forbes Sep 11 2017",
    },
    "facility:EXXON_BAYTOWN": {
        "facility": "ExxonMobil Baytown Refinery",
        "impact": "Full shutdown. 2nd largest US refinery (560k bbl/d). Shutdown Aug 27, restart began Sep 5.",
        "shutdown_days": 9,
        "severity": "EXTREME",
        "severity_rank": 2,
        "source": "EIA situation reports; Washington Post Aug 29 2017",
    },
    "facility:ARKEMA_CROSBY": {
        "facility": "Arkema Crosby Chemical Plant",
        "impact": "Catastrophic. Lost all power Aug 28, explosions Aug 31, evacuation zone to Sep 4. Most severe single-site event.",
        "shutdown_days": 10,
        "severity": "EXTREME",
        "severity_rank": 3,
        "source": "CSB investigation report; DOE situation reports",
    },
    "corridor:HOUSTON_SHIP_CHANNEL": {
        "facility": "Houston Ship Channel",
        "impact": "Closed by USCG Aug 25. Reopened with restrictions Sep 1. ~4,700 vessels delayed.",
        "shutdown_days": 7,
        "severity": "HIGH",
        "severity_rank": 4,
        "source": "USCG; WorkBoat Sep 1 2017; NOAA survey ops",
    },
    "facility:LYONDELLBASELL_CHANNELVIEW": {
        "facility": "LyondellBasell Channelview Complex",
        "impact": "Shutdown due to flooding and power loss. Estimated 7-10 day outage.",
        "shutdown_days": 8,
        "severity": "HIGH",
        "severity_rank": 5,
        "source": "Dallas Morning News; LyondellBasell press",
    },
    "facility:DOW_FREEPORT": {
        "facility": "Dow Chemical Freeport",
        "impact": "Precautionary shutdown of some units. Freeport port also closed. Estimated 5-7 day curtailment.",
        "shutdown_days": 6,
        "severity": "HIGH",
        "severity_rank": 6,
        "source": "The Guardian; Reuters industry roundup",
    },
    "facility:VALERO_HOUSTON": {
        "facility": "Valero Houston Refinery",
        "impact": "Reduced operations but NO full shutdown. 191k bbl/d facility cut rates Aug 29. Less severe than peers.",
        "shutdown_days": 3,
        "severity": "MEDIUM",
        "severity_rank": 7,
        "source": "Reuters; EIA situation reports",
    },
}

# Nodes in the blast radius that don't have direct shutdown data
# (Port Houston is the shock source, CenterPoint/ERCOT are grid-level)
SHOCK_SOURCE = "asset:PORT_HOUSTON"
EXCLUDED_FROM_RANKING = {
    "asset:PORT_HOUSTON",
    "asset:CENTERPOINT_GRID",
    "asset:ERCOT_GRID",
    "region:US-TX-HOUSTON",
}


def load_blast_radius(path: str) -> List[Dict]:
    """Load blast radius results from output.json."""
    with open(path) as f:
        data = json.load(f)
    return data.get("blast_radius", [])


def compute_spearman(predicted_ranks: List[int], observed_ranks: List[int]) -> float:
    """Compute Spearman rank correlation coefficient.

    ρ = 1 - (6 Σd²) / (n(n²-1))

    Returns value in [-1, 1]. 1.0 = perfect agreement.
    """
    n = len(predicted_ranks)
    if n < 2:
        return 0.0
    d_sq_sum = sum((p - o) ** 2 for p, o in zip(predicted_ranks, observed_ranks))
    rho = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))
    return rho


def validate(blast_radius: List[Dict], label: str = "Model") -> Dict:
    """Compare blast radius ranking to observed outcomes.

    Args:
        blast_radius: List of blast radius items (sorted by exposure desc)
        label: Label for this run (e.g., "Base", "AlphaEarth")

    Returns:
        Dict with comparison table and rank correlation
    """
    # Filter to nodes with observed outcomes
    ranked_entities = []
    for i, item in enumerate(blast_radius):
        entity = item["entity"]
        if entity in OBSERVED_OUTCOMES:
            ranked_entities.append(
                {
                    "entity": entity,
                    "predicted_rank": len(ranked_entities) + 1,
                    "predicted_exposure": item.get(
                        "total_exposure", item.get("base_exposure", 0)
                    ),
                    "predicted_tier": item.get("exposure_tier", "?"),
                }
            )

    # Add observed outcomes not in blast radius (they'd rank last)
    blast_entities = {item["entity"] for item in blast_radius}
    next_rank = len(ranked_entities) + 1
    for entity, outcome in OBSERVED_OUTCOMES.items():
        if entity not in blast_entities:
            ranked_entities.append(
                {
                    "entity": entity,
                    "predicted_rank": next_rank,
                    "predicted_exposure": 0.0,
                    "predicted_tier": "NONE",
                }
            )
            next_rank += 1

    # Attach observed data
    for item in ranked_entities:
        obs = OBSERVED_OUTCOMES.get(item["entity"], {})
        item["observed_rank"] = obs.get("severity_rank", 99)
        item["observed_days"] = obs.get("shutdown_days", 0)
        item["observed_severity"] = obs.get("severity", "?")
        item["facility"] = obs.get("facility", item["entity"])
        item["impact"] = obs.get("impact", "")
        item["rank_delta"] = abs(item["predicted_rank"] - item["observed_rank"])

    # Sort by predicted rank
    ranked_entities.sort(key=lambda x: x["predicted_rank"])

    # Compute Spearman
    pred_ranks = [item["predicted_rank"] for item in ranked_entities]
    obs_ranks = [item["observed_rank"] for item in ranked_entities]
    spearman = compute_spearman(pred_ranks, obs_ranks)

    # Agreement count
    agreements = sum(1 for item in ranked_entities if item["rank_delta"] <= 1)
    n = len(ranked_entities)

    return {
        "label": label,
        "entities": ranked_entities,
        "spearman_rho": spearman,
        "agreements": agreements,
        "total": n,
        "agreement_pct": agreements / n if n > 0 else 0,
    }


def print_validation(result: Dict):
    """Print validation results as a table."""
    entities = result["entities"]
    label = result["label"]

    print(f"\n{'=' * 96}")
    print(f"  📐 Validation: {label} vs Observed Outcomes")
    print(f"{'=' * 96}")
    print(
        f"  {'FACILITY':<30s} | {'PRED':>4s} | {'OBS':>3s} | {'|Δ|':>3s} | {'SCORE':>6s} | {'DAYS':>4s} | ACTUAL IMPACT"
    )
    print(
        f"  {'':-<30s} | {'':-<4s} | {'':-<3s} | {'':-<3s} | {'':-<6s} | {'':-<4s} | {'':-<40s}"
    )

    for item in entities:
        name = item["facility"][:30]
        pred_r = item["predicted_rank"]
        obs_r = item["observed_rank"]
        delta = item["rank_delta"]
        score = (
            f"{item['predicted_exposure']:.1%}"
            if item["predicted_exposure"] > 0
            else " N/A "
        )
        days = f"{item['observed_days']}d" if item["observed_days"] > 0 else " ?"
        impact = item["impact"][:40]

        match = "✅" if delta <= 1 else "⚠️" if delta <= 2 else "❌"
        print(
            f"  {name:<30s} | #{pred_r:<3d} | #{obs_r:<2d} | {delta:>2d}{match} | {score:>6s} | {days:>4s} | {impact}"
        )

    rho = result["spearman_rho"]
    agr = result["agreements"]
    n = result["total"]
    print(f"{'=' * 96}")
    print(
        f"  Spearman ρ = {rho:.3f}  |  Rank agreement (±1): {agr}/{n} ({result['agreement_pct']:.0%})"
    )
    print(f"{'=' * 96}\n")


def run_comparison(output_path: str):
    """Run validation comparing base vs AlphaEarth-adjusted results."""
    blast = load_blast_radius(output_path)

    # Check if this has similarity-adjusted data
    has_base = any("base_exposure" in item for item in blast)

    if has_base:
        # Build base-only blast radius (using base_exposure)
        base_blast = []
        for item in blast:
            base_item = dict(item)
            base_item["total_exposure"] = item.get(
                "base_exposure", item["total_exposure"]
            )
            base_blast.append(base_item)
        base_blast.sort(key=lambda x: x["total_exposure"], reverse=True)

        # Validate both
        base_result = validate(base_blast, label="Base Propagation (no AlphaEarth)")
        ae_result = validate(blast, label="+ Real AlphaEarth Embeddings")

        print_validation(base_result)
        print_validation(ae_result)

        # Summary comparison
        print(
            f"  📊 Spearman ρ improvement: {base_result['spearman_rho']:.3f} → {ae_result['spearman_rho']:.3f} (Δ = {ae_result['spearman_rho'] - base_result['spearman_rho']:+.3f})"
        )
        improved = ae_result["spearman_rho"] > base_result["spearman_rho"]
        print(
            f"  {'✅ AlphaEarth IMPROVES rank correlation with observed outcomes' if improved else '⚠️  AlphaEarth does NOT improve rank correlation'}\n"
        )

        return base_result, ae_result
    else:
        result = validate(blast, label="Model Output")
        print_validation(result)
        return result, None


if __name__ == "__main__":
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, "output.json")

    if not os.path.exists(output_path):
        print(
            "❌ Run the demo first: python examples/flood_replay_demo/run.py --alphaearth"
        )
    else:
        run_comparison(output_path)
