"""Run a small global target-turn attention-knockout pilot on Qwen3-14B."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen14b_attention_probe import render_with_offsets  # noqa: E402
from run_qwen14b_reproduction import LocalQwen, extract_and_evaluate  # noqa: E402


CONDITIONS = ("baseline", "direct_only", "full_downstream")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_payloads(path: Path) -> dict[str, dict]:
    return {
        row["trajectory_id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def additive_mask(
    query_positions: torch.Tensor,
    key_length: int,
    target_indices: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build an additive causal mask that also blocks target content keys."""
    keys = torch.arange(key_length, device=device)
    mask = torch.zeros((query_positions.numel(), key_length), dtype=dtype, device=device)
    mask.masked_fill_(keys.unsqueeze(0) > query_positions.unsqueeze(1), torch.finfo(dtype).min)
    valid_targets = [index for index in target_indices if index < key_length]
    if valid_targets:
        mask[:, valid_targets] = torch.finfo(dtype).min
    return mask.unsqueeze(0).unsqueeze(0)


def mask_self_test() -> None:
    mask = additive_mask(torch.tensor([3, 4]), 5, [1, 2], torch.float32, torch.device("cpu"))[0, 0]
    low = torch.finfo(torch.float32).min
    assert mask[0, 1] == low and mask[0, 2] == low
    assert mask[0, 4] == low and mask[0, 0] == 0
    assert mask[1, 1] == low and mask[1, 4] == 0


def eos_ids(tokenizer) -> set[int]:
    value = tokenizer.eos_token_id
    return set(value if isinstance(value, list) else [value])


def context_ids(engine: LocalQwen, messages: list[dict]) -> torch.Tensor:
    kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
    if engine.thinking_mode != "template_default":
        kwargs["enable_thinking"] = engine.thinking_mode == "enabled"
    ids = engine.tokenizer.apply_chat_template(messages, **kwargs)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    return ids.to(engine.model.device)


def model_step(base_model, token_ids: torch.Tensor, cache, attention_mask=None):
    return base_model(
        input_ids=token_ids,
        past_key_values=cache,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )


def initial_hidden(engine: LocalQwen, ids: torch.Tensor, target_indices: list[int], condition: str, chunk_size: int):
    base_model = engine.model.model
    dtype = next(base_model.parameters()).dtype
    device = ids.device
    if condition == "baseline":
        output = model_step(base_model, ids, None)
        hidden = output.last_hidden_state[:, -1:, :].clone()
        cache = output.past_key_values
        del output
        return hidden, cache

    if condition == "direct_only":
        if ids.shape[-1] < 2:
            raise RuntimeError("Direct-only knockout requires at least two context tokens")
        prefix = model_step(base_model, ids[:, :-1], None)
        cache = prefix.past_key_values
        del prefix
        q_position = torch.tensor([ids.shape[-1] - 1], device=device)
        mask = additive_mask(q_position, ids.shape[-1], target_indices, dtype, device)
        output = model_step(base_model, ids[:, -1:], cache, mask)
        hidden = output.last_hidden_state[:, -1:, :].clone()
        cache = output.past_key_values
        del output
        return hidden, cache

    target_end = max(target_indices) + 1
    prefix = model_step(base_model, ids[:, :target_end], None)
    cache = prefix.past_key_values
    hidden = prefix.last_hidden_state[:, -1:, :].clone()
    del prefix
    for start in range(target_end, ids.shape[-1], chunk_size):
        end = min(start + chunk_size, ids.shape[-1])
        positions = torch.arange(start, end, device=device)
        mask = additive_mask(positions, end, target_indices, dtype, device)
        output = model_step(base_model, ids[:, start:end], cache, mask)
        cache = output.past_key_values
        hidden = output.last_hidden_state[:, -1:, :]
        del output
    return hidden, cache


def generate_condition(
    engine: LocalQwen,
    messages: list[dict],
    target_indices: list[int],
    condition: str,
    chunk_size: int,
) -> tuple[str, dict]:
    ids = context_ids(engine, messages)
    max_positions = getattr(engine.model.config, "max_position_embeddings", None)
    if max_positions and ids.shape[-1] + engine.max_new_tokens > max_positions:
        raise RuntimeError("Context plus generation exceeds model position limit")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generated = []
    with torch.inference_mode():
        hidden, cache = initial_hidden(engine, ids, target_indices, condition, chunk_size)
        base_model = engine.model.model
        dtype = hidden.dtype
        device = ids.device
        for _ in range(engine.max_new_tokens):
            logits = engine.model.lm_head(hidden[:, -1:, :])[:, -1, :].float()
            token_id = int(torch.argmax(logits, dim=-1).item())
            del logits
            if token_id in eos_ids(engine.tokenizer):
                break
            generated.append(token_id)
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            if condition == "baseline":
                mask = None
            else:
                position = int(cache.get_seq_length())
                mask = additive_mask(
                    torch.tensor([position], device=device), position + 1, target_indices, dtype, device
                )
            output = model_step(base_model, token, cache, mask)
            cache = output.past_key_values
            hidden = output.last_hidden_state[:, -1:, :]
            del output
    elapsed = time.perf_counter() - started
    return engine.tokenizer.decode(generated, skip_special_tokens=True), {
        "prompt_tokens": int(ids.shape[-1]),
        "generated_tokens": len(generated),
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validate-first-baseline", action="store_true")
    args = parser.parse_args()

    mask_self_test()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot = read_csv(args.pilot_manifest)
    payloads = read_payloads(args.payload)
    samples = {row["task_id"]: row for row in json.loads(args.dataset.read_text(encoding="utf-8"))}
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = LocalQwen(args.model, args.revision, args.thinking_mode, args.max_new_tokens)

    config = {
        "stage": "S3_global_target_turn_knockout_pilot",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "conditions": list(CONDITIONS),
        "mask_scope": "target_assistant_content_tokens_only",
        "direct_only": "final_answer_decoding_queries_to_target_content",
        "full_downstream": "all_queries_after_target_content_to_target_content",
        "prefill_chunk_size": args.prefill_chunk_size,
        "model_revision": getattr(engine.model.config, "_commit_hash", None) or args.revision,
        "thinking_mode": args.thinking_mode,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "partition": os.getenv("SLURM_JOB_PARTITION"),
        "job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for row_index, row in enumerate(pilot):
        payload = payloads[row["trajectory_id"]]
        messages = payload["original_messages"][:-1]
        target_turn = int(row["target_turn"])
        span = render_with_offsets(engine.tokenizer, messages, target_turn, args.thinking_mode)
        target_indices = [
            token["token_index"] for token in span["token_rows"] if token["segment"] == "target_assistant_turn"
        ]
        if not target_indices:
            raise RuntimeError(f"Empty target span for {row['trajectory_id']}:{target_turn}")
        sample = samples[payload["source_instance_id"]]
        results = []
        for condition in CONDITIONS:
            output, metadata = generate_condition(
                engine, messages, target_indices, condition, args.prefill_chunk_size
            )
            _, evaluator, score = extract_and_evaluate(row["task"], output, sample)
            results.append((condition, output, score, evaluator, metadata))

        baseline = next(item for item in results if item[0] == "baseline")
        if row_index == 0 and args.validate_first_baseline:
            standard_output, standard_metadata = engine.generate(messages)
            _, _, standard_score = extract_and_evaluate(row["task"], standard_output, sample)
            validation = {
                "trajectory_id": row["trajectory_id"],
                "target_turn": target_turn,
                "exact_output_match": standard_output == baseline[1],
                "score_match": abs(standard_score - baseline[2]) <= 1e-12,
                "standard_score": standard_score,
                "manual_score": baseline[2],
                "standard_generation_metadata": standard_metadata,
            }
            (args.output_dir / "baseline_invariance_check.json").write_text(
                json.dumps(validation, indent=2), encoding="utf-8"
            )
            if not validation["exact_output_match"] or not validation["score_match"]:
                raise RuntimeError("Manual baseline generation differs from standard HF generation")

        append_jsonl(
            args.output_dir / "token_spans.jsonl",
            {
                "trajectory_id": row["trajectory_id"],
                "target_turn": target_turn,
                "target_token_indices": target_indices,
                "target_token_start": min(target_indices),
                "target_token_end_exclusive": max(target_indices) + 1,
                "target_token_count": len(target_indices),
                "prompt_token_count": len(span["input_ids"]),
                "mask_key_count": len(target_indices),
                "span_validation": "exact_render_offsets_and_template_id_match",
            },
        )
        for condition, output, score, evaluator, metadata in results:
            append_jsonl(
                args.output_dir / "s3_knockout_results.jsonl",
                {
                    "trajectory_id": row["trajectory_id"],
                    "target_turn": target_turn,
                    "task": row["task"],
                    "pilot_group": row["pilot_group"],
                    "reference_effect_class": row["effect_class"],
                    "condition": condition,
                    "generated_output": output,
                    "evaluator_score": score,
                    "delta_vs_baseline": score - baseline[2],
                    "target_token_start": min(target_indices),
                    "target_token_end_exclusive": max(target_indices) + 1,
                    "number_masked_keys": 0 if condition == "baseline" else len(target_indices),
                    "evaluator_output": evaluator,
                    "generation_metadata": metadata,
                },
            )
        print(json.dumps({"trajectory_id": row["trajectory_id"], "target_turn": target_turn, "status": "completed"}), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

