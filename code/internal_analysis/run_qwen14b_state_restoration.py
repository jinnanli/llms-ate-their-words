"""Patch one final-user-anchor residual state between original and surgery contexts."""
from __future__ import annotations

import argparse
import csv
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qwen14b_reproduction import LocalQwen, extract_and_evaluate  # noqa: E402
from run_qwen14b_state_propagation import render_context  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_final_anchor(engine: LocalQwen, messages: list[dict], target_turn: int) -> tuple[list[int], int]:
    input_ids, anchors = render_context(engine, messages, target_turn)
    final_user_turn = max(anchors)
    return input_ids, int(anchors[final_user_turn]["final_content_token"])


def capture_layer_inputs(
    engine: LocalQwen,
    input_ids: list[int],
    anchor_position: int,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, inputs):
            hidden = inputs[0]
            if anchor_position >= hidden.shape[1]:
                raise RuntimeError("Anchor position is invalid for source prefill")
            captured[layer] = hidden[0, anchor_position].detach().clone()
        return hook

    for layer in layers:
        handles.append(engine.model.model.layers[layer].register_forward_pre_hook(make_hook(layer)))
    tokens = torch.tensor([input_ids], dtype=torch.long, device=engine.model.device)
    try:
        with torch.inference_mode():
            output = engine.model.model(input_ids=tokens, use_cache=False, return_dict=True)
            del output
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError(f"Missing captured layers: {sorted(set(layers) - set(captured))}")
    return captured


def generate_with_patch(
    engine: LocalQwen,
    input_ids: list[int],
    anchor_position: int,
    layer: int,
    source_state: torch.Tensor,
) -> tuple[str, dict]:
    applied = 0

    def patch_hook(_module, inputs):
        nonlocal applied
        hidden = inputs[0]
        if applied or hidden.shape[1] <= anchor_position:
            return None
        patched = hidden.clone()
        patched[0, anchor_position] = source_state.to(device=hidden.device, dtype=hidden.dtype)
        applied += 1
        return (patched, *inputs[1:])

    handle = engine.model.model.layers[layer].register_forward_pre_hook(patch_hook)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=engine.model.device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = engine.model.generate(
                tokens,
                max_new_tokens=engine.max_new_tokens,
                do_sample=False,
                pad_token_id=engine.tokenizer.pad_token_id or engine.tokenizer.eos_token_id,
                eos_token_id=engine.tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    if applied != 1:
        raise RuntimeError(f"Patch at layer {layer} applied {applied} times")
    new_tokens = generated[0, tokens.shape[-1] :]
    return engine.tokenizer.decode(new_tokens, skip_special_tokens=True), {
        "prompt_tokens": int(tokens.shape[-1]),
        "generated_tokens": int(new_tokens.shape[-1]),
        "elapsed_seconds": time.perf_counter() - started,
        "patch_apply_count": applied,
        "anchor_position": anchor_position,
    }


def validate_messages(original: list[dict], surgery: list[dict], target_turn: int) -> None:
    original_users = [row["content"] for row in original if row["role"] == "user"]
    surgery_users = [row["content"] for row in surgery if row["role"] == "user"]
    if original_users != surgery_users:
        raise RuntimeError("Original and surgery fixed user messages differ")
    if target_turn >= len(original_users) - 1:
        raise RuntimeError("Target turn is not an eligible intermediate turn")


def evaluate(task: str, output: str, sample: dict) -> tuple[float, dict]:
    _, evaluator, score = extract_and_evaluate(task, output, sample)
    return score, evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,6,11,17,22,28,33,39")
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--sham-only", action="store_true")
    parser.add_argument("--sham-trajectories-per-task", type=int, default=1)
    parser.add_argument("--restoration-mode", choices=["positive", "negative"], default="positive")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(value) for value in args.layers.replace(":", ",").split(",")]
    rows = read_csv(args.pairs)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)
    pair_ids = sorted(grouped)
    selected_ids = [pair_id for index, pair_id in enumerate(pair_ids) if index % args.shard_count == args.shard_index]
    selected = [row for pair_id in selected_ids for row in grouped[pair_id]]

    baselines = {row["trajectory_id"]: row for row in read_jsonl(args.native_dir / "server_native_baselines.jsonl")}
    surgeries = {
        (row["trajectory_id"], int(row["target_turn"])): row
        for row in read_jsonl(args.native_dir / "server_native_surgeries.jsonl")
    }
    for row in selected:
        key = (row["trajectory_id"], int(row["target_turn"]))
        if row["trajectory_id"] not in baselines or key not in surgeries:
            raise RuntimeError(f"Missing native input for {key}")
        original = baselines[row["trajectory_id"]]["generated_history"][:-1]
        surgery = surgeries[key]["surgery_history"][:-1]
        validate_messages(original, surgery, key[1])
    if args.validate_inputs_only:
        (args.output_dir / "input_validation.json").write_text(
            json.dumps({"pairs": len(selected_ids), "branches": len(selected), "status": "PASS"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = LocalQwen(args.model, args.revision, args.thinking_mode, args.max_new_tokens)
    if layers != sorted(set(layers)) or layers[0] < 0 or layers[-1] >= engine.model.config.num_hidden_layers:
        raise RuntimeError(f"Invalid layer scan {layers} for {engine.model.config.num_hidden_layers} layers")
    samples = {row["task_id"]: row for row in json.loads(args.dataset.read_text(encoding="utf-8"))}

    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "population_version": selected[0]["population_version"],
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "model": args.model,
        "model_revision": getattr(engine.model.config, "_commit_hash", None) or args.revision,
        "torch_version": torch.__version__,
        "layers": layers,
        "patch_site": "decoder_layer_input_final_fixed_user_content_token",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pair_count": len(selected_ids),
        "branch_count": len(selected),
        "sham_only": args.sham_only,
        "restoration_mode": args.restoration_mode,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    if args.sham_only:
        sham_rows = []
        task_seen = defaultdict(int)
        for pair_id in selected_ids:
            positive = next(row for row in grouped[pair_id] if row["sample_role"] == "positive")
            if task_seen[positive["task"]] < args.sham_trajectories_per_task:
                sham_rows.append(positive)
                task_seen[positive["task"]] += 1
        if len(task_seen) != 3:
            raise RuntimeError(f"Sham validation does not cover all tasks: {dict(task_seen)}")
        for row in sham_rows:
            trajectory_id = row["trajectory_id"]
            target_turn = int(row["target_turn"])
            baseline = baselines[trajectory_id]
            surgery_row = surgeries[(trajectory_id, target_turn)]
            original_messages = baseline["generated_history"][:-1]
            surgery_messages = surgery_row["surgery_history"][:-1]
            original_ids, original_anchor = render_final_anchor(engine, original_messages, target_turn)
            surgery_ids, surgery_anchor = render_final_anchor(engine, surgery_messages, target_turn)
            original_states = capture_layer_inputs(engine, original_ids, original_anchor, layers)
            surgery_states = capture_layer_inputs(engine, surgery_ids, surgery_anchor, layers)
            sample = samples[baseline["source_instance_id"]]
            for context_name, ids, anchor, states, expected_output, expected_score in (
                ("original", original_ids, original_anchor, original_states, baseline["original_output_local"], float(baseline["original_score_local"])),
                ("surgery", surgery_ids, surgery_anchor, surgery_states, surgery_row["surgery_output_local"], float(surgery_row["surgery_score_local"])),
            ):
                for layer in layers:
                    output, metadata = generate_with_patch(engine, ids, anchor, layer, states[layer])
                    score, evaluator = evaluate(row["task"], output, sample)
                    exact_output = output == expected_output
                    exact_score = abs(score - expected_score) <= 1e-12
                    append_jsonl(args.output_dir / "sham_validation.jsonl", {
                        "pair_id": row["pair_id"], "trajectory_id": trajectory_id, "task": row["task"],
                        "target_turn": target_turn, "context": context_name, "layer": layer,
                        "exact_output_match": exact_output, "exact_score_match": exact_score,
                        "expected_score": expected_score, "patched_score": score,
                        "evaluator": evaluator, "metadata": metadata,
                    })
                    if not exact_output or not exact_score:
                        raise RuntimeError(f"Sham invariance failed for {trajectory_id}/{context_name}/layer={layer}")
        (args.output_dir / "SHAM_VALIDATION_PASS").write_text("PASS\n", encoding="ascii")
        return

    original_cache: dict[str, tuple[list[int], int, dict[int, torch.Tensor]]] = {}
    for row in selected:
        trajectory_id = row["trajectory_id"]
        target_turn = int(row["target_turn"])
        baseline = baselines[trajectory_id]
        surgery_row = surgeries[(trajectory_id, target_turn)]
        original_messages = baseline["generated_history"][:-1]
        surgery_messages = surgery_row["surgery_history"][:-1]
        if trajectory_id not in original_cache:
            original_ids, original_anchor = render_final_anchor(engine, original_messages, target_turn)
            original_cache[trajectory_id] = (
                original_ids,
                original_anchor,
                capture_layer_inputs(engine, original_ids, original_anchor, layers),
            )
        original_ids, original_anchor, original_states = original_cache[trajectory_id]
        surgery_ids, surgery_anchor = render_final_anchor(engine, surgery_messages, target_turn)
        surgery_states = capture_layer_inputs(engine, surgery_ids, surgery_anchor, layers)
        sample = samples[baseline["source_instance_id"]]
        for layer in layers:
            directions = (
                ("rescue", original_ids, original_anchor, surgery_states[layer]),
                ("reversion", surgery_ids, surgery_anchor, original_states[layer]),
            ) if args.restoration_mode == "positive" else (
                ("harm_transfer", original_ids, original_anchor, surgery_states[layer]),
                ("rescue_reversal", surgery_ids, surgery_anchor, original_states[layer]),
            )
            for direction, destination_ids, destination_anchor, source_state in directions:
                output, metadata = generate_with_patch(
                    engine, destination_ids, destination_anchor, layer, source_state
                )
                score, evaluator = evaluate(row["task"], output, sample)
                append_jsonl(args.output_dir / "state_restoration_results.jsonl", {
                    "pair_id": row["pair_id"], "trajectory_id": trajectory_id, "task": row["task"],
                    "sample_role": row["sample_role"], "local_effect_class": row["local_effect_class"],
                    "target_turn": target_turn, "relative_position": float(row["relative_position"]),
                    "pair_relative_position_gap": float(row["pair_relative_position_gap"]),
                    "patch_direction": direction, "layer": layer,
                    "original_score": float(row["original_score"]), "surgery_score": float(row["surgery_score"]),
                    "patched_score": score, "local_delta": float(row["local_delta"]),
                    "rescue_effect": score - float(row["original_score"]) if direction == "rescue" else None,
                    "reversion_effect": float(row["surgery_score"]) - score if direction == "reversion" else None,
                    "harm_transfer_effect": float(row["original_score"]) - score if direction == "harm_transfer" else None,
                    "rescue_reversal_effect": score - float(row["surgery_score"]) if direction == "rescue_reversal" else None,
                    "generated_output": output, "evaluator": evaluator, "metadata": metadata,
                    "population_version": row["population_version"],
                })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected = len(selected) * len(layers) * 2
    actual = sum(1 for _ in (args.output_dir / "state_restoration_results.jsonl").open(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError(f"Expected {expected} result rows, found {actual}")
    (args.output_dir / "COMPLETE").write_text(f"rows={actual}\n", encoding="ascii")


if __name__ == "__main__":
    main()

