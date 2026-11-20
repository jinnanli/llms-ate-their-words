"""Technical Qwen3-14B target-turn attention probe for the S1 gate.

This script is intentionally narrow. It reuses the Stage S0 replay and
evaluation path, generates with SDPA, and extracts attention post hoc with a
cached, one-token-at-a-time forward pass. It never persists full attention
matrices.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qwen14b_reproduction import (  # noqa: E402
    LocalQwen,
    extract_and_evaluate,
    read_jsonl,
    replay,
)


INTERFACE_VERSION = "mechanistic_interface_v1"
LONG_FIELDS = [
    "mechanistic_interface_version",
    "trajectory_id",
    "target_turn",
    "task",
    "condition",
    "layer",
    "head",
    "query_region",
    "target_attention_raw",
    "target_attention_normalized",
    "assistant_history_attention",
    "final_user_attention",
    "n_target_tokens",
    "n_assistant_history_tokens",
    "n_final_user_tokens",
    "n_generated_tokens",
    "n_query_tokens",
]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_with_offsets(tokenizer, messages: list[dict], target_turn: int, thinking_mode: str) -> dict:
    """Render once, then map exact tokenizer offsets to message content spans."""
    template_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if thinking_mode != "template_default":
        template_kwargs["enable_thinking"] = thinking_mode == "enabled"
    rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]

    token_template_kwargs = dict(template_kwargs)
    token_template_kwargs["tokenize"] = True
    templated_ids = tokenizer.apply_chat_template(messages, **token_template_kwargs)
    if hasattr(templated_ids, "tolist"):
        templated_ids = templated_ids.tolist()
    if templated_ids and isinstance(templated_ids[0], list):
        templated_ids = templated_ids[0]
    if list(templated_ids) != input_ids:
        raise RuntimeError("Rendered-offset token IDs differ from apply_chat_template token IDs")

    last_user_message = max(i for i, message in enumerate(messages) if message["role"] == "user")
    content_spans = []
    cursor = 0
    assistant_turn = -1
    for message_index, message in enumerate(messages):
        content = message["content"]
        start = rendered.find(content, cursor)
        if start < 0:
            raise RuntimeError(f"Could not locate rendered content for message {message_index}")
        end = start + len(content)
        cursor = end
        assistant_index = None
        if message["role"] == "assistant":
            assistant_turn += 1
            assistant_index = assistant_turn
            segment = "target_assistant_turn" if assistant_turn == target_turn else "other_assistant_history"
        elif message["role"] == "user":
            segment = "final_user_turn" if message_index == last_user_message else "user_history"
        else:
            segment = "system_template"
        content_spans.append(
            {
                "start_char": start,
                "end_char": end,
                "role": message["role"],
                "message_index": message_index,
                "assistant_turn_index": assistant_index,
                "is_target_turn": assistant_index == target_turn,
                "segment": segment,
            }
        )

    token_rows = []
    for token_index, (token_id, (start, end)) in enumerate(zip(input_ids, offsets)):
        owner = None
        if end > start:
            overlaps = [span for span in content_spans if start < span["end_char"] and end > span["start_char"]]
            if len(overlaps) > 1:
                raise RuntimeError(f"Token {token_index} crosses multiple message-content boundaries")
            if overlaps:
                span = overlaps[0]
                if start < span["start_char"] or end > span["end_char"]:
                    raise RuntimeError(f"Token {token_index} crosses a template/content boundary")
                owner = span
        token_rows.append(
            {
                "token_index": token_index,
                "token_id": int(token_id),
                "token_text": tokenizer.convert_ids_to_tokens(int(token_id)),
                "role": owner["role"] if owner else "template",
                "message_index": owner["message_index"] if owner else None,
                "assistant_turn_index": owner["assistant_turn_index"] if owner else None,
                "is_target_turn": bool(owner and owner["is_target_turn"]),
                "segment": owner["segment"] if owner else "system_template",
                "start_char": int(start),
                "end_char": int(end),
            }
        )

    required = {"target_assistant_turn", "final_user_turn"}
    observed = {row["segment"] for row in token_rows}
    missing = required - observed
    if missing:
        raise RuntimeError(f"Required token segments are empty: {sorted(missing)}")

    # Compact contiguous intervals are sufficient to reproduce every key mask.
    intervals = []
    for row in token_rows:
        identity = (
            row["segment"],
            row["role"],
            row["message_index"],
            row["assistant_turn_index"],
            row["is_target_turn"],
        )
        if intervals and intervals[-1]["identity"] == identity:
            intervals[-1]["end_token"] = row["token_index"] + 1
        else:
            intervals.append(
                {
                    "identity": identity,
                    "start_token": row["token_index"],
                    "end_token": row["token_index"] + 1,
                    "segment": row["segment"],
                    "role": row["role"],
                    "message_index": row["message_index"],
                    "assistant_turn_index": row["assistant_turn_index"],
                    "is_target_turn": row["is_target_turn"],
                }
            )
    for interval in intervals:
        interval.pop("identity")

    segment_counts = defaultdict(int)
    for row in token_rows:
        segment_counts[row["segment"]] += 1
    return {
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "token_rows": token_rows,
        "intervals": intervals,
        "segment_counts": dict(segment_counts),
    }


def generate_with_ids(engine: LocalQwen, messages: list[dict]) -> tuple[str, list[int], dict]:
    """Use the exact Stage S0 deterministic SDPA generation settings."""
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if engine.thinking_mode != "template_default":
        template_kwargs["enable_thinking"] = engine.thinking_mode == "enabled"
    input_ids = engine.tokenizer.apply_chat_template(messages, **template_kwargs)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(engine.model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = engine.model.generate(
            input_ids,
            max_new_tokens=engine.max_new_tokens,
            do_sample=False,
            pad_token_id=engine.tokenizer.pad_token_id or engine.tokenizer.eos_token_id,
            eos_token_id=engine.tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    new_tokens = generated[0, input_ids.shape[-1] :].detach().cpu().tolist()
    text = engine.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, new_tokens, {"prompt_tokens": int(input_ids.shape[-1]), "elapsed_seconds": elapsed}


def content_token_ids(tokenizer, generated_ids: list[int]) -> list[int]:
    """Exclude the terminal EOS decision from generated-content summaries."""
    eos_ids = tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    content = []
    for token_id in generated_ids:
        if token_id in eos_ids:
            break
        content.append(int(token_id))
    return content


def aggregate_cached_attention(
    engine: LocalQwen,
    context_ids: list[int],
    generated_ids: list[int],
    token_rows: list[dict],
    first_k: int,
) -> tuple[list[dict], dict]:
    """SDPA-prefill context, then aggregate eager one-token attention online."""
    if len(context_ids) < 2:
        raise RuntimeError("Context must contain at least two tokens for cached extraction")
    generated_ids = content_token_ids(engine.tokenizer, generated_ids)
    if not generated_ids:
        raise RuntimeError("Generation contained no non-special content tokens")

    target_indices = [row["token_index"] for row in token_rows if row["segment"] == "target_assistant_turn"]
    assistant_indices = [
        row["token_index"]
        for row in token_rows
        if row["segment"] in {"target_assistant_turn", "other_assistant_history"}
    ]
    final_user_indices = [row["token_index"] for row in token_rows if row["segment"] == "final_user_turn"]
    if not target_indices or not assistant_indices or not final_user_indices:
        raise RuntimeError("One or more required key regions are empty")

    device = engine.model.device
    context = torch.tensor([context_ids], dtype=torch.long, device=device)
    accumulators: dict[str, dict[str, Any]] = {
        "first_token": {"count": 0, "metrics": None},
        f"first_{first_k}": {"count": 0, "metrics": None},
        "all_generated": {"count": 0, "metrics": None},
    }
    base_model = engine.model.model
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        # Long-context prefill remains SDPA and does not request attention.
        prefill = base_model(
            input_ids=context[:, :-1],
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        cache = prefill.past_key_values
        del prefill

        # The final context token predicts generated token 0. Subsequent
        # generated token t predicts token t+1. Each attention query has q=1.
        step_inputs = [int(context_ids[-1]), *generated_ids[:-1]]
        for generated_index, token_id in enumerate(step_inputs):
            step = torch.tensor([[token_id]], dtype=torch.long, device=device)
            outputs = base_model(
                input_ids=step,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            layer_metrics = []
            for attention in outputs.attentions:
                if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[2] != 1:
                    raise RuntimeError(f"Unexpected attention shape: {tuple(attention.shape)}")
                metrics = torch.stack(
                    [
                        attention[0, :, 0, target_indices].sum(dim=-1),
                        attention[0, :, 0, assistant_indices].sum(dim=-1),
                        attention[0, :, 0, final_user_indices].sum(dim=-1),
                    ],
                    dim=-1,
                ).float().cpu()
                layer_metrics.append(metrics)
            metrics_tensor = torch.stack(layer_metrics, dim=0)
            regions = ["all_generated"]
            if generated_index == 0:
                regions.append("first_token")
            if generated_index < first_k:
                regions.append(f"first_{first_k}")
            for region in regions:
                accumulator = accumulators[region]
                accumulator["metrics"] = (
                    metrics_tensor.clone()
                    if accumulator["metrics"] is None
                    else accumulator["metrics"] + metrics_tensor
                )
                accumulator["count"] += 1
            del outputs, layer_metrics, metrics_tensor

    elapsed = time.perf_counter() - started
    rows = []
    for query_region, accumulator in accumulators.items():
        count = int(accumulator["count"])
        if count == 0:
            continue
        mean_metrics = accumulator["metrics"] / count
        for layer in range(mean_metrics.shape[0]):
            for head in range(mean_metrics.shape[1]):
                target_raw = float(mean_metrics[layer, head, 0])
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_region": query_region,
                        "target_attention_raw": target_raw,
                        "target_attention_normalized": target_raw / len(target_indices),
                        "assistant_history_attention": float(mean_metrics[layer, head, 1]),
                        "final_user_attention": float(mean_metrics[layer, head, 2]),
                        "n_target_tokens": len(target_indices),
                        "n_assistant_history_tokens": len(assistant_indices),
                        "n_final_user_tokens": len(final_user_indices),
                        "n_generated_tokens": len(generated_ids),
                        "n_query_tokens": count,
                    }
                )
    metadata = {
        "attention_shapes": [
            int(engine.model.config.num_hidden_layers),
            int(engine.model.config.num_attention_heads),
            1,
            "context_plus_generated_prefix",
        ],
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "prefill_backend": engine.model.config._attn_implementation,
        "attention_backend": "per-call eager fallback via output_attentions=True",
        "full_attention_persisted": False,
    }
    return rows, metadata


def case_summary(long_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in long_rows:
        grouped[row["query_region"]].append(row)
    summaries = []
    for region, rows in grouped.items():
        summaries.append(
            {
                "query_region": region,
                "mean_target_attention_raw": sum(row["target_attention_raw"] for row in rows) / len(rows),
                "mean_target_attention_normalized": sum(row["target_attention_normalized"] for row in rows) / len(rows),
                "mean_assistant_history_attention": sum(row["assistant_history_attention"] for row in rows) / len(rows),
                "mean_final_user_attention": sum(row["final_user_attention"] for row in rows) / len(rows),
                "layer_head_cells": len(rows),
                "n_query_tokens": rows[0]["n_query_tokens"],
            }
        )
    return summaries


def select_probe_rows(rows: list[dict], args) -> list[tuple[dict, str]]:
    branches = [row for row in rows if row.get("condition") == "turn_surgery"]
    if args.trajectory_id:
        wanted = set(args.trajectory_id)
        branches = [row for row in branches if row["trajectory_id"] in wanted]
    if args.target_turn:
        wanted_turns = set(args.target_turn)
        branches = [row for row in branches if int(row["target_turn"]) in wanted_turns]
    selected = [(row, condition) for row in branches for condition in args.condition]
    return selected[: args.max_cases]


def messages_for_span_only(row: dict, condition: str) -> list[dict]:
    messages = copy.deepcopy(row["original_messages"][:-1])
    if condition == "surgery":
        assistant_index = -1
        for message in messages:
            if message["role"] == "assistant":
                assistant_index += 1
                if assistant_index == int(row["target_turn"]):
                    message["content"] = "Acknowledged."
    return messages


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=Path,
        default=ROOT / "mechanistic_interface/v1/qwen14b_reproduction_subset.jsonl",
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/sharded_instructions_600.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-id", action="append", default=[])
    parser.add_argument("--target-turn", action="append", type=int, default=[])
    parser.add_argument("--condition", action="append", choices=["original", "surgery"], required=True)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--first-k", type=int, default=8)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--span-only", action="store_true")
    parser.add_argument("--save-token-map", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to write into non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_probe_rows(read_jsonl(args.subset), args)
    if not rows:
        raise RuntimeError("No matching target-turn branches were selected")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.span_only:
        engine = None
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        samples = {}
    else:
        engine = LocalQwen(args.model, args.revision, args.thinking_mode, args.max_new_tokens)
        tokenizer = engine.tokenizer
        samples = {row["task_id"]: row for row in json.loads(args.dataset.read_text(encoding="utf-8"))}
    run_config = {
        "stage": "S1_implementation_preparation",
        "technical_probe_only": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_argument": args.model,
        "revision_argument": args.revision,
        "resolved_model_commit": (
            getattr(engine.model.config, "_commit_hash", None)
            if engine is not None
            else tokenizer.init_kwargs.get("_commit_hash")
        ),
        "thinking_mode": args.thinking_mode,
        "generation_backend": "sdpa",
        "attention_method": "sdpa_prefill_then_cached_output_attentions_eager_fallback",
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "first_k": args.first_k,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and engine is not None else None,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    all_long_rows = []
    all_case_summaries = []
    for row, condition in rows:
        target_turn = int(row["target_turn"])
        identity = {
            "mechanistic_interface_version": INTERFACE_VERSION,
            "trajectory_id": row["trajectory_id"],
            "target_turn": target_turn,
            "task": row["task"],
            "condition": condition,
        }
        try:
            if args.span_only:
                messages = messages_for_span_only(row, condition)
                baseline_output = None
                baseline_score = None
            else:
                replay_row = copy.deepcopy(row)
                replay_row["condition"] = "sharded_replay" if condition == "original" else "turn_surgery"
                history, generation_calls = replay(replay_row, engine)
                messages = history[:-1]
                baseline_output = history[-1]["content"]
                sample = samples[row["source_instance_id"]]
                _, _, baseline_score = extract_and_evaluate(row["task"], baseline_output, sample)

            span = render_with_offsets(tokenizer, messages, target_turn, args.thinking_mode)
            span_record = {
                **identity,
                "input_token_count": len(span["input_ids"]),
                "segment_counts": span["segment_counts"],
                "intervals": span["intervals"],
                "validation": "exact_render_offsets_and_template_id_match",
            }
            append_jsonl(args.output_dir / "token_spans.jsonl", span_record)
            if args.save_token_map:
                append_jsonl(args.output_dir / "token_maps.jsonl", {**identity, "tokens": span["token_rows"]})

            if args.span_only:
                append_jsonl(args.output_dir / "probe_results.jsonl", {**identity, "status": "span_only_pass"})
                continue

            attention_output, generated_ids, generation_meta = generate_with_ids(engine, messages)
            sample = samples[row["source_instance_id"]]
            _, _, attention_score = extract_and_evaluate(row["task"], attention_output, sample)
            long_rows, attention_meta = aggregate_cached_attention(
                engine,
                span["input_ids"],
                generated_ids,
                span["token_rows"],
                args.first_k,
            )
            long_rows = [{**identity, **attention_row} for attention_row in long_rows]
            all_long_rows.extend(long_rows)
            summaries = [{**identity, **summary} for summary in case_summary(long_rows)]
            all_case_summaries.extend(summaries)
            append_jsonl(
                args.output_dir / "probe_results.jsonl",
                {
                    **identity,
                    "status": "completed",
                    "baseline_generated_output": baseline_output,
                    "attention_mode_generated_output": attention_output,
                    "exact_output_match": baseline_output == attention_output,
                    "baseline_evaluator_score": baseline_score,
                    "attention_mode_evaluator_score": attention_score,
                    "score_match": abs(float(baseline_score) - float(attention_score)) <= 1e-12,
                    "generation_metadata": generation_meta,
                    "attention_metadata": attention_meta,
                    "span_segment_counts": span["segment_counts"],
                    "generation_calls": generation_calls,
                },
            )
        except Exception as error:
            append_jsonl(
                args.output_dir / "probe_results.jsonl",
                {
                    **identity,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    if all_long_rows:
        write_csv(args.output_dir / "attention_layer_head.csv", all_long_rows, LONG_FIELDS)
        summary_fields = list(all_case_summaries[0].keys())
        write_csv(args.output_dir / "attention_case_summary.csv", all_case_summaries, summary_fields)


if __name__ == "__main__":
    main()

