"""Score server-native success/failure continuations under exact native contexts."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qwen14b_reproduction import LocalQwen  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_manifest(path: Path) -> dict[tuple[str, int], dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["trajectory_id"], int(row["target_turn"])): row
            for row in csv.DictReader(handle)
        }


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_context_ids(engine: LocalQwen, messages: list[dict]) -> torch.Tensor:
    kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
    if engine.thinking_mode != "template_default":
        kwargs["enable_thinking"] = engine.thinking_mode == "enabled"
    ids = engine.tokenizer.apply_chat_template(messages, **kwargs)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    return ids.to(engine.model.device)


def continuation_ids(engine: LocalQwen, text: str) -> list[int]:
    ids = engine.tokenizer.encode(text, add_special_tokens=False)
    decoded = engine.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != text:
        raise RuntimeError("Continuation does not round-trip through tokenizer exactly")
    if not ids:
        raise RuntimeError("Empty continuation")
    return ids


def sequence_loglik(engine: LocalQwen, messages: list[dict], text: str, chunk_size: int = 32) -> dict:
    """Teacher-force content tokens only; EOS is explicitly excluded."""
    context = build_context_ids(engine, messages)
    target_ids = continuation_ids(engine, text)
    target = torch.tensor(target_ids, dtype=torch.long, device=engine.model.device)
    base_model = engine.model.model
    total = 0.0
    with torch.inference_mode():
        context_out = base_model(input_ids=context, use_cache=True, output_attentions=False, return_dict=True)
        cache = context_out.past_key_values
        first_hidden = context_out.last_hidden_state[:, -1:, :]
        del context_out
        if len(target_ids) > 1:
            continuation_out = base_model(
                input_ids=target[:-1].unsqueeze(0),
                past_key_values=cache,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )
            hidden = torch.cat([first_hidden, continuation_out.last_hidden_state], dim=1)[0]
            del continuation_out
        else:
            hidden = first_hidden[0]
        for start in range(0, len(target_ids), chunk_size):
            end = min(start + chunk_size, len(target_ids))
            logits = engine.model.lm_head(hidden[start:end]).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            total += float(log_probs.gather(1, target[start:end].unsqueeze(1)).sum())
            del logits, log_probs
    return {
        "sum_log_likelihood": total,
        "mean_log_likelihood": total / len(target_ids),
        "token_count": len(target_ids),
        "eos_included": False,
        "prompt_token_count": int(context.shape[-1]),
    }


def select_population(baselines: list[dict], surgeries: list[dict], manifest: dict) -> tuple[list[dict], list[dict]]:
    baseline_by_id = {row["trajectory_id"]: row for row in baselines if row["status"] == "completed"}
    enriched = []
    for row in surgeries:
        key = (row["trajectory_id"], int(row["target_turn"]))
        if row["status"] != "completed" or key not in manifest or row["trajectory_id"] not in baseline_by_id:
            continue
        metadata = manifest[key]
        enriched.append(
            {
                **row,
                "relative_position": float(metadata["relative_position"]),
                "n_total_turns": int(metadata["n_total_turns"]),
            }
        )
    primary = [
        row for row in enriched
        if row["task"] in {"actions", "code"}
        and float(row["original_score_local"]) == 0.0
        and float(row["surgery_score_local"]) == 1.0
    ]
    zeros = [row for row in enriched if abs(float(row["delta_local"])) <= 1e-12]
    controls = []
    used = set()
    for row in sorted(primary, key=lambda item: (item["task"], item["trajectory_id"], item["target_turn"])):
        candidates = [
            candidate for candidate in zeros
            if candidate["task"] == row["task"]
            and (candidate["trajectory_id"], candidate["target_turn"]) not in used
        ]
        if not candidates:
            continue
        control = min(
            candidates,
            key=lambda item: (
                abs(item["relative_position"] - row["relative_position"]),
                abs(item["n_total_turns"] - row["n_total_turns"]),
                item["trajectory_id"], item["target_turn"],
            ),
        )
        used.add((control["trajectory_id"], control["target_turn"]))
        controls.append(control)
    return primary, controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baselines = read_jsonl(args.native_dir / "server_native_baselines.jsonl")
    surgeries = read_jsonl(args.native_dir / "server_native_surgeries.jsonl")
    manifest = read_manifest(args.manifest)
    primary, controls = select_population(baselines, surgeries, manifest)
    population = [("fail_to_success", row) for row in primary] + [("matched_nochange", row) for row in controls]

    engine = LocalQwen(args.model, args.revision, args.thinking_mode, max_new_tokens=1)
    config = {
        "stage": "S2_native_contrastive_probability",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "native_dir": str(args.native_dir),
        "primary_count": len(primary),
        "control_count": len(controls),
        "model_revision": getattr(engine.model.config, "_commit_hash", None) or args.revision,
        "thinking_mode": args.thinking_mode,
        "continuation_scoring": "content_tokens_only_teacher_forced",
        "eos_included": False,
        "partition": os.getenv("SLURM_JOB_PARTITION"),
        "job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    baseline_by_id = {row["trajectory_id"]: row for row in baselines}
    for group, row in population:
        baseline = baseline_by_id[row["trajectory_id"]]
        c_orig = baseline["generated_history"][:-1]
        c_surg = row["surgery_history"][:-1]
        y_failure = row["original_output_local"]
        y_success = row["surgery_output_local"]
        success_orig = sequence_loglik(engine, c_orig, y_success)
        failure_orig = sequence_loglik(engine, c_orig, y_failure)
        success_surg = sequence_loglik(engine, c_surg, y_success)
        failure_surg = sequence_loglik(engine, c_surg, y_failure)
        margin_orig = success_orig["mean_log_likelihood"] - failure_orig["mean_log_likelihood"]
        margin_surg = success_surg["mean_log_likelihood"] - failure_surg["mean_log_likelihood"]
        append_jsonl(
            args.output_dir / "s2_sequence_scores.jsonl",
            {
                "trajectory_id": row["trajectory_id"],
                "target_turn": row["target_turn"],
                "task": row["task"],
                "s2_group": group,
                "effect_class_local": row["effect_class_local"],
                "ll_success_original_context": success_orig,
                "ll_failure_original_context": failure_orig,
                "ll_success_surgery_context": success_surg,
                "ll_failure_surgery_context": failure_surg,
                "margin_original": margin_orig,
                "margin_surgery": margin_surg,
                "delta_margin": margin_surg - margin_orig,
            },
        )
        print(json.dumps({"trajectory_id": row["trajectory_id"], "target_turn": row["target_turn"], "status": "completed"}), flush=True)


if __name__ == "__main__":
    main()

