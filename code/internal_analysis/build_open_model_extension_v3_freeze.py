"""Build the final, non-manuscript v3 evidence handoff from frozen analyses."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLAIMS = [
    ("Influential surgery produces greater downstream state divergence", "SUPPORTED_WITH_CAVEAT", "Task-macro and within-trajectory State Propagation are positive; pooled estimates are task-composition sensitive."),
    ("Downstream state divergence is task-structured", "SUPPORTED", "Task-specific and task-macro estimates differ materially; pooled signs can be misleading."),
    ("Replacing beneficial-surgery states with original-history states can partially undo benefit", "SUPPORTED_WITH_CAVEAT", "Formal and prospective D2T reversion are directionally positive; effects are sparse and uncertain."),
    ("The D2T reversion restoration effect independently replicates", "SUPPORTED_WITH_CAVEAT", "Prospective predefined regions are positive, with 40-layer profile agreement, but CIs cross zero."),
    ("Restoration is bidirectional", "NOT_SUPPORTED", "Rescue is null or opposite while reversion is positive."),
    ("Restoration is signed-symmetric", "NOT_SUPPORTED", "Signed-negative harm transfer is zero and rescue reversal is not coherent."),
    ("Restoration generalizes across Actions, Code, and D2T", "NOT_SUPPORTED", "Actions and Code restoration are zero across all formal layers."),
    ("Restoration is localized to a reproducible depth region", "SUPPORTED_WITH_CAVEAT", "Prospective bands overlap formal regions, but replication is directional rather than precise."),
    ("State Restoration mediates Turn Surgery", "NOT_SUPPORTED", "No mediation analysis or complete causal decomposition was performed."),
    ("State Propagation identifies the unique causal mechanism", "NOT_SUPPORTED", "Profile correspondence is descriptive and alternative pathways remain possible."),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-analysis", type=Path, required=True)
    parser.add_argument("--v2-number-lookup", type=Path, required=True)
    parser.add_argument("--signed-negative", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = read_csv(args.v3_analysis / "STATE_RESTORATION_FULL_LAYER_REPLICATION.csv")
    formal_results = read_csv(args.v3_analysis / "formal_full_layer_results.csv")
    prospective_results = read_csv(args.v3_analysis / "prospective_full_layer_results.csv")
    correlations = read_csv(args.v3_analysis / "profile_correlations.csv")
    integrity = json.loads((args.v3_analysis / "integrity.json").read_text(encoding="utf-8"))
    depth = json.loads((args.v3_analysis / "depth_profile_replication.json").read_text(encoding="utf-8"))
    v2 = read_csv(args.v2_number_lookup)
    signed = read_csv(args.signed_negative)

    if len(comparison) != 43 or integrity["prospective_full_layer"]["status"] != "PASS":
        raise RuntimeError("Final full-layer analysis is incomplete")

    state_rows = []
    state_selectors = [
        ("FORMAL_HELDOUT_118", "State final anchor", "task_macro", None, "ASSOCIATED_STATE_DIVERGENCE", "TASK_STRUCTURED"),
        ("FORMAL_HELDOUT_118", "State AUC", "task_macro", None, "ASSOCIATED_STATE_DIVERGENCE", "TASK_STRUCTURED"),
        ("FORMAL_HELDOUT_118", "State Propagation within-trajectory", "all", None, "ASSOCIATED_STATE_DIVERGENCE", "WITHIN_TRAJECTORY_REPLICATED"),
        ("PROSPECTIVE_D2T_117", "State Propagation", "D2T", "final_anchor_divergence", "ASSOCIATED_STATE_DIVERGENCE", "PROSPECTIVE_D2T_POSITIVE"),
        ("PROSPECTIVE_D2T_117", "State Propagation", "D2T", "propagation_auc", "ASSOCIATED_STATE_DIVERGENCE", "PROSPECTIVE_D2T_NULL_AUC"),
    ]
    for population, experiment, scope, metric, evidence_type, status in state_selectors:
        matches = [
            row for row in v2
            if row["population"] == population and row["experiment"] == experiment
            and row["task_scope"] == scope and (metric is None or row["metric"] == metric)
        ]
        for row in matches:
            probe = experiment
            if experiment == "State Propagation within-trajectory":
                probe = f"State Propagation within-trajectory {'final anchor' if 'final_' in row['metric'] else 'AUC'}"
            elif population == "PROSPECTIVE_D2T_117":
                probe = f"State Propagation {'final anchor' if 'final_' in row['metric'] else 'AUC'}"
            state_rows.append({
                "evidence_type": evidence_type,
                "population": population,
                "task": scope,
                "probe": probe,
                "direction": "positive_minus_zero",
                "layer_or_region": scope,
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "trajectory_n": row["trajectory_n"],
                "pair_n": "193" if experiment == "State Propagation within-trajectory" and scope == "all" else "",
                "replication_status": status,
                "interpretation_boundary": "Association, not mediation",
            })

    region_lookup = {row["layer_or_region"]: row for row in comparison if row["summary"] == "region"}
    for population, prefix, n, status in (
        ("FORMAL_HELDOUT_D2T", "formal", 33, "FORMAL_TASK_SPECIFIC_SUPPORT"),
        ("PROSPECTIVE_D2T", "prospective", 43, "DIRECTIONAL_FULL_LAYER_REPLICATION"),
    ):
        for region in ("3-12", "16-31", "3-12_union_16-31"):
            row = region_lookup[region]
            for direction in ("rescue", "reversion"):
                state_rows.append({
                    "evidence_type": "CAUSAL_STATE_INTERVENTION_EFFECT",
                    "population": population,
                    "task": "data2text",
                    "probe": "State Restoration",
                    "direction": direction,
                    "layer_or_region": region,
                    "estimate": row[f"{prefix}_{direction}_difference"],
                    "ci_low": row[f"{prefix}_{direction}_ci_low"],
                    "ci_high": row[f"{prefix}_{direction}_ci_high"],
                    "trajectory_n": n,
                    "pair_n": n,
                    "replication_status": status,
                    "interpretation_boundary": "Causal state intervention; not mediation",
                })
    write_csv(args.output_dir / "STATE_PROPAGATION_RESTORATION_SYNTHESIS.csv", state_rows)

    figure = []
    for row in state_rows:
        priority = "CORE_CANDIDATE" if row["probe"] == "State Restoration" and row["direction"] == "reversion" or row["probe"].startswith("State Propagation within-trajectory") else "SPACE_DEPENDENT"
        figure.append({
            "population": row["population"], "task": row["task"], "probe": row["probe"],
            "direction": row["direction"], "layer_or_region": row["layer_or_region"],
            "estimate": row["estimate"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "trajectory_n": row["trajectory_n"], "pair_n": row["pair_n"],
            "replication_status": row["replication_status"], "display_priority": priority,
        })
    for population, rows, n, status in (
        ("FORMAL_HELDOUT_D2T", formal_results, 33, "FORMAL_TASK_SPECIFIC_SUPPORT"),
        ("PROSPECTIVE_D2T", prospective_results, 43, "DIRECTIONAL_FULL_LAYER_REPLICATION"),
    ):
        for row in rows:
            if row["summary"] != "layer" or row["scope"] != "data2text":
                continue
            direction = row["direction"]
            figure.append({
                "population": population, "task": "data2text", "probe": "State Restoration",
                "direction": direction, "layer_or_region": row["layer"],
                "estimate": row["difference"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "trajectory_n": n, "pair_n": n, "replication_status": status,
                "display_priority": "CORE_CANDIDATE" if direction == "reversion" else "SPACE_DEPENDENT",
            })
    signed_layer17 = next(row for row in signed if row["direction"] == "rescue_reversal" and row["depth"] == "layer" and row["layer"] == "17" and row["scope"] == "data2text")
    figure.append({
        "population": "FORMAL_HELDOUT_NEGATIVE", "task": "data2text", "probe": "Signed-negative restoration",
        "direction": "rescue_reversal", "layer_or_region": "17", "estimate": signed_layer17["difference"],
        "ci_low": signed_layer17["ci_low"], "ci_high": signed_layer17["ci_high"], "trajectory_n": 23,
        "pair_n": 23, "replication_status": "NO_SIGNED_RESTORATION_SIGNAL",
        "display_priority": "DO_NOT_INCLUDE_IN_SUBMISSION",
    })
    write_csv(args.output_dir / "OPEN_MODEL_EXTENSION_V3_FIGURE_DATA.csv", figure)

    number_rows = []
    for row in figure:
        number_rows.append({
            "population": row["population"], "task": row["task"], "experiment": row["probe"],
            "metric": row["direction"], "layer_or_region": row["layer_or_region"], "estimate": row["estimate"],
            "ci_low": row["ci_low"], "ci_high": row["ci_high"], "trajectory_n": row["trajectory_n"],
            "pair_n": row["pair_n"], "support_status": "SUPPORTED_WITH_CAVEAT" if row["direction"] == "reversion" or row["probe"] != "State Restoration" else "NOT_SUPPORTED",
            "page_priority": row["display_priority"],
        })
    for row in correlations:
        if row["comparison"] == "formal_vs_prospective":
            number_rows.append({
                "population": "FORMAL_VS_PROSPECTIVE", "task": "data2text", "experiment": "State Restoration profile",
                "metric": f"{row['direction']}_spearman", "layer_or_region": "40_layers", "estimate": row["spearman"],
                "ci_low": "", "ci_high": "", "trajectory_n": "33/43", "pair_n": "33/43",
                "support_status": "SUPPORTED_WITH_CAVEAT", "page_priority": "CORE_CANDIDATE",
            })
    write_csv(args.output_dir / "OPEN_MODEL_EXTENSION_V3_NUMBER_LOOKUP.csv", number_rows)

    claim_lines = ["# Open-Model Extension v3 Claim Lookup", ""]
    for claim, status, reason in CLAIMS:
        claim_lines.extend((f"## {claim}", f"**{status}**", "", reason, ""))
    (args.output_dir / "OPEN_MODEL_EXTENSION_V3_CLAIM_LOOKUP.md").write_text("\n".join(claim_lines), encoding="utf-8")

    reversion = depth["reversion"]
    handoff = f"""# Open-Model Extension v3 Handoff

## Scope
- Population A: `FORMAL_HELDOUT_D2T`, 33 deterministic positive-zero pairs.
- Population B: `PROSPECTIVE_D2T`, 43 deterministic positive-zero pairs.
- Patch site: final fixed-user-anchor residual state.
- Directions: rescue and reversion, analyzed separately.

## Integrity
- Formal full layer: `{integrity['formal_full_layer']['actual_rows']}/{integrity['formal_full_layer']['expected_rows']}`, `{integrity['formal_full_layer']['status']}`.
- Prospective full layer: `{integrity['prospective_full_layer']['actual_rows']}/{integrity['prospective_full_layer']['expected_rows']}`, `{integrity['prospective_full_layer']['status']}`.

## Primary Evidence Card
- Probe: D2T reversion State Restoration.
- Replication: `DIRECTIONAL_FULL_LAYER_REPLICATION`.
- 40-layer Spearman: `{reversion['spearman_40_layers']}`.
- Sign concordance: `{reversion['sign_concordance_all']}/40`.
- Region A concordance: `{reversion['sign_concordance_regions']['3-12']}/10`.
- Region B concordance: `{reversion['sign_concordance_regions']['16-31']}/16`.
- Longest prospective positive band: `{reversion['prospective_longest_positive_band']}`.
- Boundary: directional, D2T-specific, reversion-only; no mediation claim.

## Evidence Hierarchy
- `PRIMARY_V3_FINDING`: task-balanced/within-trajectory State Propagation plus D2T reversion causal follow-up.
- `SECONDARY_V3_FINDING`: prospective full-layer directional replication.
- `SUPPORTING_EVIDENCE`: task-specific S1b, D2T S3, S2 margin signal.
- `BOUNDARY_RESULTS`: rescue unsupported; Actions/Code zero; signed-negative null.
- `DO_NOT_EMPHASIZE`: predictive analysis and profile correspondence as standalone evidence.

## Freeze
- `OPEN_MODEL_EXTENSION_V3 = FINAL_FROZEN`
- `ADDITIONAL_V3_COMPUTE = NO`
- `V2_FINAL_FREEZE_UNCHANGED = YES`
- `MANUSCRIPT_REVIEW_REQUIRED = YES`
"""
    (args.output_dir / "OPEN_MODEL_EXTENSION_V3_HANDOFF.md").write_text(handoff, encoding="utf-8")

    validation = {
        "status": "PASS",
        "formal_integrity": integrity["formal_full_layer"],
        "prospective_integrity": integrity["prospective_full_layer"],
        "full_layer_rows": len([row for row in comparison if row["summary"] == "layer"]),
        "region_rows": len([row for row in comparison if row["summary"] == "region"]),
        "figure_rows": len(figure),
        "number_rows": len(number_rows),
        "claim_count": len(CLAIMS),
        "classification": "DIRECTIONAL_FULL_LAYER_REPLICATION",
    }
    (args.output_dir / "FINAL_FREEZE_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

