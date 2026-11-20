"""Measure residual-stream divergence at fixed downstream user anchors."""
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen14b_attention_probe import render_with_offsets  # noqa: E402
from run_qwen14b_reproduction import LocalQwen  # noqa: E402


LAYER_BANDS = {
    "early": range(0, 10),
    "mid_early": range(10, 20),
    "mid_late": range(20, 30),
    "late": range(30, 40),
}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def user_anchor_map(token_rows: list[dict], messages: list[dict]) -> dict[int, dict]:
    user_turn = -1
    message_to_turn = {}
    for message_index, message in enumerate(messages):
        if message["role"] == "user":
            user_turn += 1
            message_to_turn[message_index] = user_turn
    grouped = defaultdict(list)
    for token in token_rows:
        if token["role"] == "user" and token["message_index"] in message_to_turn:
            grouped[message_to_turn[token["message_index"]]].append(token["token_index"])
    result = {}
    for message_index, turn in message_to_turn.items():
        indices = grouped[turn]
        if not indices:
            raise RuntimeError(f"User message {turn} has no content tokens")
        result[turn] = {
            "message_index": message_index,
            "content": messages[message_index]["content"],
            "content_token_indices": indices,
            "final_content_token": max(indices),
        }
    return result


def render_context(engine: LocalQwen, messages: list[dict], target_turn: int) -> tuple[list[int], dict[int, dict]]:
    rendered = render_with_offsets(engine.tokenizer, messages, target_turn, engine.thinking_mode)
    return rendered["input_ids"], user_anchor_map(rendered["token_rows"], messages)


def extract_anchor_states(
    engine: LocalQwen,
    input_ids: list[int],
    anchors: dict[int, dict],
    wanted_turns: list[int],
    chunk_size: int,
) -> dict[int, torch.Tensor]:
    """Advance with cache and retain only one residual vector per layer/anchor."""
    base_model = engine.model.model
    device = engine.model.device
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    cursor = 0
    cache = None
    states = {}
    with torch.inference_mode():
        for user_turn in sorted(wanted_turns, key=lambda turn: anchors[turn]["final_content_token"]):
            anchor_position = anchors[user_turn]["final_content_token"]
            while cursor < anchor_position:
                end = min(cursor + chunk_size, anchor_position)
                output = base_model(
                    input_ids=tokens[:, cursor:end],
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
                cache = output.past_key_values
                cursor = end
                del output
            output = base_model(
                input_ids=tokens[:, cursor : cursor + 1],
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            cache = output.past_key_values
            if len(output.hidden_states) != engine.model.config.num_hidden_layers + 1:
                raise RuntimeError("Unexpected hidden-state layer count")
            states[user_turn] = torch.stack(
                [hidden[0, -1].detach().float().cpu() for hidden in output.hidden_states[1:]], dim=0
            )
            cursor += 1
            del output
    return states


def layer_band(layer: int) -> str:
    return next(name for name, layers in LAYER_BANDS.items() if layer in layers)


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def normalized_auc(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    step = 1.0 / (len(values) - 1)
    return step * (sum(values) - 0.5 * values[0] - 0.5 * values[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-branches", type=int)
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    if args.max_branches is not None:
        manifest = manifest[: args.max_branches]
    baselines = {row["trajectory_id"]: row for row in read_jsonl(args.native_dir / "server_native_baselines.jsonl")}
    surgeries = {
        (row["trajectory_id"], int(row["target_turn"])): row
        for row in read_jsonl(args.native_dir / "server_native_surgeries.jsonl")
    }
    if args.validate_inputs_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        adapter = type("TokenizerAdapter", (), {"tokenizer": tokenizer, "thinking_mode": args.thinking_mode})()
        checks = []
        for row in manifest:
            key = (row["trajectory_id"], int(row["target_turn"]))
            original_messages = baselines[key[0]]["generated_history"][:-1]
            surgery_messages = surgeries[key]["surgery_history"][:-1]
            original_ids, original_anchors = render_context(adapter, original_messages, key[1])
            surgery_ids, surgery_anchors = render_context(adapter, surgery_messages, key[1])
            wanted = list(range(key[1] + 1, int(row["n_user_turns"])))
            exact = all(
                original_anchors[turn]["content"] == surgery_anchors[turn]["content"]
                for turn in wanted
            )
            if not exact:
                raise RuntimeError(f"Fixed user anchor mismatch for {key}")
            checks.append({
                "trajectory_id": key[0], "target_turn": key[1],
                "downstream_anchors": len(wanted), "fixed_user_content_match": exact,
                "original_prompt_tokens": len(original_ids), "surgery_prompt_tokens": len(surgery_ids),
            })
        (args.output_dir / "input_validation.json").write_text(
            json.dumps({"status": "PASS", "branches": len(checks), "checks": checks}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"status": "PASS", "branches": len(checks)}))
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = LocalQwen(args.model, args.revision, args.thinking_mode, max_new_tokens=1)
    run_config = {
        "stage": "state_propagation_fixed_user_anchors_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "branch_count": len(manifest),
        "anchor_representation": "final_user_content_token",
        "hidden_states": "decoder_layers_only_excluding_embedding",
        "distance": "one_minus_cosine",
        "chunk_size": args.chunk_size,
        "model_revision": getattr(engine.model.config, "_commit_hash", None) or args.revision,
        "thinking_mode": args.thinking_mode,
        "dtype": "bfloat16_forward_float32_cosine",
        "partition": os.getenv("SLURM_JOB_PARTITION"),
        "job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    grouped = defaultdict(list)
    for row in manifest:
        grouped[row["trajectory_id"]].append(row)
    for trajectory_id, branch_rows in sorted(grouped.items()):
        original_messages = baselines[trajectory_id]["generated_history"][:-1]
        original_ids, original_anchors = render_context(
            engine, original_messages, min(int(row["target_turn"]) for row in branch_rows)
        )
        all_wanted = sorted({turn for row in branch_rows for turn in range(int(row["target_turn"]) + 1, int(row["n_user_turns"]))})
        original_states = extract_anchor_states(
            engine, original_ids, original_anchors, all_wanted, args.chunk_size
        )
        for row in sorted(branch_rows, key=lambda item: int(item["target_turn"])):
            key = (trajectory_id, int(row["target_turn"]))
            surgery_messages = surgeries[key]["surgery_history"][:-1]
            surgery_ids, surgery_anchors = render_context(engine, surgery_messages, key[1])
            wanted = list(range(key[1] + 1, int(row["n_user_turns"])))
            for turn in wanted:
                if original_anchors[turn]["content"] != surgery_anchors[turn]["content"]:
                    raise RuntimeError(f"Fixed user anchor mismatch for {key}, user turn {turn}")
            surgery_states = extract_anchor_states(
                engine, surgery_ids, surgery_anchors, wanted, args.chunk_size
            )
            anchor_rows = []
            for ordinal, turn in enumerate(wanted):
                distances = 1 - F.cosine_similarity(
                    original_states[turn], surgery_states[turn], dim=-1
                )
                progress = 1.0 if len(wanted) == 1 else ordinal / (len(wanted) - 1)
                for layer, distance in enumerate(distances.tolist()):
                    anchor_row = {
                        "population_version": row.get(
                            "population_version", "DISCOVERY_FORMAL_ALIGNED_NATIVE_24"
                        ),
                        "source_shard": row.get("source_shard", "discovery"),
                        "trajectory_id": trajectory_id,
                        "target_turn": key[1],
                        "task": row["task"],
                        "local_effect_class": row["local_effect_class"],
                        "local_delta": float(row["local_delta"]),
                        "formal_effect_class": row["formal_effect_class"],
                        "downstream_user_turn": turn,
                        "downstream_anchor_ordinal": ordinal,
                        "downstream_anchor_count": len(wanted),
                        "normalized_downstream_progress": progress,
                        "layer": layer,
                        "layer_band": layer_band(layer),
                        "cosine_distance": float(distance),
                        "original_token_position": original_anchors[turn]["final_content_token"],
                        "surgery_token_position": surgery_anchors[turn]["final_content_token"],
                        "token_position_shift": (
                            surgery_anchors[turn]["final_content_token"]
                            - original_anchors[turn]["final_content_token"]
                        ),
                        "user_content_sha256": content_hash(original_anchors[turn]["content"]),
                    }
                    anchor_rows.append(anchor_row)
                    append_jsonl(args.output_dir / "state_propagation_anchor_level.jsonl", anchor_row)
            for band_name in ["all_layers", *LAYER_BANDS]:
                selected = anchor_rows if band_name == "all_layers" else [
                    item for item in anchor_rows if item["layer_band"] == band_name
                ]
                by_anchor = defaultdict(list)
                for item in selected:
                    by_anchor[item["downstream_user_turn"]].append(item["cosine_distance"])
                profile = [mean_or_none(by_anchor[turn]) for turn in wanted]
                summary_row = {
                    "population_version": row.get(
                        "population_version", "DISCOVERY_FORMAL_ALIGNED_NATIVE_24"
                    ),
                    "source_shard": row.get("source_shard", "discovery"),
                    "trajectory_id": trajectory_id,
                    "target_turn": key[1],
                    "task": row["task"],
                    "local_effect_class": row["local_effect_class"],
                    "local_delta": float(row["local_delta"]),
                    "formal_effect_class": row["formal_effect_class"],
                    "layer_group": band_name,
                    "n_downstream_anchors": len(wanted),
                    "final_user_anchor_divergence": profile[-1],
                    "mean_downstream_anchor_divergence": mean_or_none(profile),
                    "propagation_auc": normalized_auc(profile),
                    "normalized_propagation_profile": profile,
                }
                append_jsonl(args.output_dir / "state_propagation_branch_summary.jsonl", summary_row)
            print(json.dumps({"trajectory_id": trajectory_id, "target_turn": key[1], "status": "completed"}), flush=True)

    for source_name, target_name in (
        ("state_propagation_anchor_level.jsonl", "state_propagation_anchor_level.csv"),
        ("state_propagation_branch_summary.jsonl", "state_propagation_branch_summary.csv"),
    ):
        rows = read_jsonl(args.output_dir / source_name)
        with (args.output_dir / target_name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

