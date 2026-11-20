"""Measure native original-to-surgery attention redistribution on Qwen3-14B."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen14b_attention_probe import content_token_ids, generate_with_ids, render_with_offsets  # noqa: E402
from run_qwen14b_reproduction import LocalQwen  # noqa: E402


LAYER_BANDS = {
    "early": range(0, 10),
    "mid_early": range(10, 20),
    "mid_late": range(20, 30),
    "late": range(30, 40),
}
LONG_FIELDS = [
    "population_version", "source_shard", "trajectory_id", "target_turn", "task",
    "local_effect_class", "local_delta", "formal_effect_class",
    "query_region", "layer", "head", "original_final_user_attention", "surgery_final_user_attention",
    "delta_final_user_attention", "original_assistant_history_attention", "surgery_assistant_history_attention",
    "delta_assistant_history_attention", "original_assistant_history_attention_normalized",
    "surgery_assistant_history_attention_normalized", "delta_assistant_history_attention_normalized",
    "original_previous_user_attention", "surgery_previous_user_attention", "delta_previous_user_attention",
    "n_original_assistant_tokens", "n_surgery_assistant_tokens", "n_final_user_tokens",
    "n_original_previous_user_tokens", "n_surgery_previous_user_tokens", "n_original_generated_tokens",
    "n_surgery_generated_tokens", "n_query_tokens_original", "n_query_tokens_surgery",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def context_attention(
    engine: LocalQwen,
    context_ids: list[int],
    generated_ids: list[int],
    token_rows: list[dict],
    first_k: int,
) -> tuple[list[dict], dict]:
    generated_ids = content_token_ids(engine.tokenizer, generated_ids)
    if len(context_ids) < 2 or not generated_ids:
        raise RuntimeError("Context or generated continuation is empty")
    assistant = [row["token_index"] for row in token_rows if row["role"] == "assistant"]
    final_user = [row["token_index"] for row in token_rows if row["segment"] == "final_user_turn"]
    previous_user = [
        row["token_index"] for row in token_rows if row["role"] == "user" and row["segment"] != "final_user_turn"
    ]
    if not assistant or not final_user:
        raise RuntimeError("Assistant-history or final-user key span is empty")
    regions = ("first_token", f"first_{first_k}", "all_generated")
    accumulators = {region: {"count": 0, "value": None} for region in regions}
    context = torch.tensor([context_ids], dtype=torch.long, device=engine.model.device)
    base_model = engine.model.model
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        prefill = base_model(
            input_ids=context[:, :-1], use_cache=True, output_attentions=False, return_dict=True
        )
        cache = prefill.past_key_values
        del prefill
        step_inputs = [int(context_ids[-1]), *generated_ids[:-1]]
        for generated_index, token_id in enumerate(step_inputs):
            outputs = base_model(
                input_ids=torch.tensor([[token_id]], dtype=torch.long, device=engine.model.device),
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            layer_values = []
            for attention in outputs.attentions:
                if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[2] != 1:
                    raise RuntimeError(f"Unexpected attention shape: {tuple(attention.shape)}")
                weights = attention[0, :, 0, :]
                prior = weights[:, previous_user].sum(dim=-1) if previous_user else torch.zeros_like(weights[:, 0])
                layer_values.append(
                    torch.stack(
                        [
                            weights[:, final_user].sum(dim=-1),
                            weights[:, assistant].sum(dim=-1),
                            prior,
                        ],
                        dim=-1,
                    ).float().cpu()
                )
            values = torch.stack(layer_values, dim=0)
            active = ["all_generated"]
            if generated_index == 0:
                active.append("first_token")
            if generated_index < first_k:
                active.append(f"first_{first_k}")
            for region in active:
                accumulator = accumulators[region]
                accumulator["value"] = values.clone() if accumulator["value"] is None else accumulator["value"] + values
                accumulator["count"] += 1
            del outputs, layer_values, values
    rows = []
    for region in regions:
        count = accumulators[region]["count"]
        values = accumulators[region]["value"] / count
        for layer in range(values.shape[0]):
            for head in range(values.shape[1]):
                rows.append(
                    {
                        "query_region": region,
                        "layer": layer,
                        "head": head,
                        "final_user_attention": float(values[layer, head, 0]),
                        "assistant_history_attention": float(values[layer, head, 1]),
                        "assistant_history_attention_normalized": float(values[layer, head, 1]) / len(assistant),
                        "previous_user_attention": float(values[layer, head, 2]),
                        "n_assistant_tokens": len(assistant),
                        "n_final_user_tokens": len(final_user),
                        "n_previous_user_tokens": len(previous_user),
                        "n_generated_tokens": len(generated_ids),
                        "n_query_tokens": count,
                    }
                )
    return rows, {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "full_attention_persisted": False,
        "attention_shape": [engine.model.config.num_hidden_layers, engine.model.config.num_attention_heads, 1, "growing_k"],
    }


def combine_rows(metadata: dict, original: list[dict], surgery: list[dict]) -> list[dict]:
    original_by_key = {(row["query_region"], row["layer"], row["head"]): row for row in original}
    surgery_by_key = {(row["query_region"], row["layer"], row["head"]): row for row in surgery}
    if set(original_by_key) != set(surgery_by_key):
        raise RuntimeError("Original and surgery attention grids differ")
    output = []
    for key in sorted(original_by_key):
        before = original_by_key[key]
        after = surgery_by_key[key]
        row = {
            **metadata,
            "query_region": key[0], "layer": key[1], "head": key[2],
            "original_final_user_attention": before["final_user_attention"],
            "surgery_final_user_attention": after["final_user_attention"],
            "delta_final_user_attention": after["final_user_attention"] - before["final_user_attention"],
            "original_assistant_history_attention": before["assistant_history_attention"],
            "surgery_assistant_history_attention": after["assistant_history_attention"],
            "delta_assistant_history_attention": after["assistant_history_attention"] - before["assistant_history_attention"],
            "original_assistant_history_attention_normalized": before["assistant_history_attention_normalized"],
            "surgery_assistant_history_attention_normalized": after["assistant_history_attention_normalized"],
            "delta_assistant_history_attention_normalized": (
                after["assistant_history_attention_normalized"] - before["assistant_history_attention_normalized"]
            ),
            "original_previous_user_attention": before["previous_user_attention"],
            "surgery_previous_user_attention": after["previous_user_attention"],
            "delta_previous_user_attention": after["previous_user_attention"] - before["previous_user_attention"],
            "n_original_assistant_tokens": before["n_assistant_tokens"],
            "n_surgery_assistant_tokens": after["n_assistant_tokens"],
            "n_final_user_tokens": before["n_final_user_tokens"],
            "n_original_previous_user_tokens": before["n_previous_user_tokens"],
            "n_surgery_previous_user_tokens": after["n_previous_user_tokens"],
            "n_original_generated_tokens": before["n_generated_tokens"],
            "n_surgery_generated_tokens": after["n_generated_tokens"],
            "n_query_tokens_original": before["n_query_tokens"],
            "n_query_tokens_surgery": after["n_query_tokens"],
        }
        output.append(row)
    return output


def summarize(rows: list[dict], layer_group: str, layers: set[int] | None) -> list[dict]:
    selected = rows if layers is None else [row for row in rows if row["layer"] in layers]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["query_region"]].append(row)
    metrics = [
        "original_final_user_attention", "surgery_final_user_attention", "delta_final_user_attention",
        "original_assistant_history_attention", "surgery_assistant_history_attention", "delta_assistant_history_attention",
        "original_assistant_history_attention_normalized", "surgery_assistant_history_attention_normalized",
        "delta_assistant_history_attention_normalized", "original_previous_user_attention",
        "surgery_previous_user_attention", "delta_previous_user_attention",
    ]
    output = []
    for query_region, values in grouped.items():
        summary = {
            key: values[0][key]
            for key in (
                "population_version", "source_shard", "trajectory_id", "target_turn", "task",
                "local_effect_class", "local_delta", "formal_effect_class",
            )
        }
        summary.update({"query_region": query_region, "layer_group": layer_group, "layer_head_cells": len(values)})
        summary.update({metric: sum(row[metric] for row in values) / len(values) for metric in metrics})
        output.append(summary)
    return output


def validate_inputs(tokenizer, thinking_mode: str, manifest: list[dict], baselines: dict, surgeries: dict) -> dict:
    checks = []
    for row in manifest:
        key = (row["trajectory_id"], int(row["target_turn"]))
        original_messages = baselines[key[0]]["generated_history"][:-1]
        surgery_messages = surgeries[key]["surgery_history"][:-1]
        before = render_with_offsets(tokenizer, original_messages, key[1], thinking_mode)
        after = render_with_offsets(tokenizer, surgery_messages, key[1], thinking_mode)
        before_final = next(message["content"] for message in reversed(original_messages) if message["role"] == "user")
        after_final = next(message["content"] for message in reversed(surgery_messages) if message["role"] == "user")
        if before_final != after_final:
            raise RuntimeError(f"Final fixed user content differs for {key}")
        checks.append(
            {
                "trajectory_id": key[0], "target_turn": key[1],
                "original_prompt_tokens": len(before["input_ids"]), "surgery_prompt_tokens": len(after["input_ids"]),
                "final_user_content_exact_match": True,
            }
        )
    return {"status": "PASS", "branches": len(checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--first-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--validate-first-pair", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.manifest)
    unique_keys = {(row["trajectory_id"], row["target_turn"]) for row in manifest}
    if not manifest or len(unique_keys) != len(manifest):
        raise RuntimeError("S1b manifest must contain non-empty unique branch keys")
    baselines = {row["trajectory_id"]: row for row in read_jsonl(args.native_dir / "server_native_baselines.jsonl")}
    surgeries = {
        (row["trajectory_id"], int(row["target_turn"])): row
        for row in read_jsonl(args.native_dir / "server_native_surgeries.jsonl")
    }
    if args.validate_inputs_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        validation = validate_inputs(tokenizer, args.thinking_mode, manifest, baselines, surgeries)
        (args.output_dir / "input_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps({"status": validation["status"], "branches": validation["branches"]}))
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = LocalQwen(args.model, args.revision, args.thinking_mode, args.max_new_tokens)
    config = {
        "stage": "S1b_counterfactual_attention_redistribution_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "native_dir": str(args.native_dir), "branch_count": len(manifest),
        "original_attention_policy": "once_per_trajectory_reused_across_branches",
        "surgery_attention_policy": "once_per_branch_exact_native_surgery_history",
        "query_regions": ["first_token", f"first_{args.first_k}", "all_generated"],
        "key_regions": ["final_user_content", "all_previous_assistant_content", "previous_user_content"],
        "model_revision": getattr(engine.model.config, "_commit_hash", None) or args.revision,
        "thinking_mode": args.thinking_mode, "do_sample": False, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "dtype": "bfloat16",
        "partition": os.getenv("SLURM_JOB_PARTITION"), "job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    grouped = defaultdict(list)
    for row in manifest:
        grouped[row["trajectory_id"]].append(row)
    first_pair = True
    for trajectory_id, branch_rows in sorted(grouped.items()):
        original_messages = baselines[trajectory_id]["generated_history"][:-1]
        original_reference = baselines[trajectory_id]["original_output_local"]
        first_target = int(branch_rows[0]["target_turn"])
        original_span = render_with_offsets(engine.tokenizer, original_messages, first_target, args.thinking_mode)
        original_output, original_ids, original_generation = generate_with_ids(engine, original_messages)
        original_attention, original_extraction = context_attention(
            engine, original_span["input_ids"], original_ids, original_span["token_rows"], args.first_k
        )
        for row in sorted(branch_rows, key=lambda item: int(item["target_turn"])):
            key = (trajectory_id, int(row["target_turn"]))
            surgery_messages = surgeries[key]["surgery_history"][:-1]
            surgery_reference = surgeries[key]["surgery_output_local"]
            surgery_span = render_with_offsets(engine.tokenizer, surgery_messages, key[1], args.thinking_mode)
            surgery_output, surgery_ids, surgery_generation = generate_with_ids(engine, surgery_messages)
            surgery_attention, surgery_extraction = context_attention(
                engine, surgery_span["input_ids"], surgery_ids, surgery_span["token_rows"], args.first_k
            )
            original_final_user = next(message["content"] for message in reversed(original_messages) if message["role"] == "user")
            surgery_final_user = next(message["content"] for message in reversed(surgery_messages) if message["role"] == "user")
            if original_final_user != surgery_final_user:
                raise RuntimeError(f"Final fixed user content differs for {key}")
            validation = {
                "population_version": row.get(
                    "population_version", "DISCOVERY_FORMAL_ALIGNED_NATIVE_24"
                ),
                "source_shard": row.get("source_shard", "discovery"),
                "trajectory_id": trajectory_id, "target_turn": key[1],
                "original_exact_native_output_match": original_output == original_reference,
                "surgery_exact_native_output_match": surgery_output == surgery_reference,
                "final_user_content_exact_match": True,
                "original_generation": original_generation, "surgery_generation": surgery_generation,
                "original_extraction": original_extraction, "surgery_extraction": surgery_extraction,
            }
            if first_pair and args.validate_first_pair:
                (args.output_dir / "first_pair_invariance_check.json").write_text(
                    json.dumps(validation, indent=2), encoding="utf-8"
                )
                if not validation["original_exact_native_output_match"] or not validation["surgery_exact_native_output_match"]:
                    raise RuntimeError("First S1b pair does not reproduce native deterministic outputs")
                first_pair = False
            append_jsonl(args.output_dir / "probe_results.jsonl", validation)
            append_jsonl(
                args.output_dir / "token_spans.jsonl",
                {
                    "population_version": row.get(
                        "population_version", "DISCOVERY_FORMAL_ALIGNED_NATIVE_24"
                    ),
                    "source_shard": row.get("source_shard", "discovery"),
                    "trajectory_id": trajectory_id, "target_turn": key[1],
                    "original_segment_counts": original_span["segment_counts"],
                    "surgery_segment_counts": surgery_span["segment_counts"],
                    "original_prompt_tokens": len(original_span["input_ids"]),
                    "surgery_prompt_tokens": len(surgery_span["input_ids"]),
                },
            )
            metadata = {
                "population_version": row.get("population_version", "DISCOVERY_FORMAL_ALIGNED_NATIVE_24"),
                "source_shard": row.get("source_shard", "discovery"),
                "trajectory_id": trajectory_id, "target_turn": key[1], "task": row["task"],
                "local_effect_class": row["local_effect_class"], "local_delta": float(row["local_delta"]),
                "formal_effect_class": row["formal_effect_class"],
            }
            combined = combine_rows(metadata, original_attention, surgery_attention)
            append_csv(args.output_dir / "s1b_layer_head.csv", combined, LONG_FIELDS)
            branch_summary = summarize(combined, "all_layers", None)
            append_csv(args.output_dir / "s1b_branch_summary.csv", branch_summary, list(branch_summary[0]))
            band_rows = []
            for band, layers in LAYER_BANDS.items():
                band_rows.extend(summarize(combined, band, set(layers)))
            append_csv(args.output_dir / "s1b_layer_band_summary.csv", band_rows, list(band_rows[0]))
            print(json.dumps({"trajectory_id": trajectory_id, "target_turn": key[1], "status": "completed"}), flush=True)


if __name__ == "__main__":
    main()

