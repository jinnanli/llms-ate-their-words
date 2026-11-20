"""Replay a frozen Qwen3-14B reproduction subset with a local HF checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))


ACK = "Acknowledged."
BINARY_TASKS = {"actions", "code", "database", "math"}


def get_task(task_name: str):
    """Load only evaluators represented in the server-only gate subset."""
    if task_name == "actions":
        from tasks.actions.task_actions import TaskActions

        return TaskActions()
    if task_name == "code":
        from tasks.code.task_code import TaskCode

        return TaskCode()
    if task_name == "data2text":
        from tasks.data2text.task_data2text import TaskData2Text

        return TaskData2Text()
    raise RuntimeError(f"Task {task_name} is outside the server-only evaluator subset")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def effect_class(delta: float, tolerance: float = 1e-12) -> str:
    if delta > tolerance:
        return "positive"
    if delta < -tolerance:
        return "negative"
    return "zero"


def extract_and_evaluate(task_name: str, output: str, sample: dict) -> tuple[str, dict, float]:
    task = get_task(task_name)
    if task.answer_extraction_strategy == "full_response":
        answer = output
    elif task.answer_extraction_strategy == "task_specific":
        answer = task.extract_answer(output)
    else:
        raise RuntimeError(
            f"{task_name} requires the formal external answer extractor; excluded from this server-only gate"
        )
    result = task.evaluator_function(answer, sample)
    score = result.get("score")
    if score is None and isinstance(result.get("is_correct"), bool):
        score = float(result["is_correct"])
    if score is None:
        raise RuntimeError(f"Evaluator returned no score for {task_name}")
    return answer, result, float(score)


class LocalQwen:
    def __init__(self, model_path: str, revision: str | None, thinking_mode: str, max_new_tokens: int):
        self.model_path = model_path
        self.revision = revision
        self.thinking_mode = thinking_mode
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=False,
        ).eval()

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        template_kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        if self.thinking_mode != "template_default":
            template_kwargs["enable_thinking"] = self.thinking_mode == "enabled"
        input_ids = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions and input_ids.shape[-1] + self.max_new_tokens > max_positions:
            raise RuntimeError(f"Context exceeds checkpoint limit: {input_ids.shape[-1]} + {self.max_new_tokens} > {max_positions}")
        input_ids = input_ids.to(self.model.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        new_tokens = generated[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, {
            "prompt_tokens": int(input_ids.shape[-1]),
            "generated_tokens": int(new_tokens.shape[-1]),
            "elapsed_seconds": elapsed,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        }


def replay(row: dict, engine: LocalQwen) -> tuple[list[dict], list[dict]]:
    original = row["original_messages"]
    system = next(message for message in original if message["role"] == "system")
    users = [message["content"] for message in original if message["role"] == "user"]
    assistants = [message["content"] for message in original if message["role"] == "assistant"]
    if len(users) != len(assistants):
        raise RuntimeError("Frozen payload does not contain one assistant turn per user turn")
    history = [system]
    calls = []
    target = row.get("target_turn")
    for turn, user in enumerate(users):
        history.append({"role": "user", "content": user})
        if row["condition"] == "turn_surgery" and turn < int(target):
            assistant = assistants[turn]
        elif row["condition"] == "turn_surgery" and turn == int(target):
            assistant = ACK
        else:
            assistant, metadata = engine.generate(history)
            calls.append({"turn_index": turn, **metadata})
        history.append({"role": "assistant", "content": assistant})
    return history, calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-reference-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "reproduction_results.jsonl"
    if result_path.exists() and not args.resume:
        raise RuntimeError(f"Refusing to overwrite existing results: {result_path}")
    existing_results = read_jsonl(result_path) if result_path.exists() else []
    completed = {row["reproduction_id"] for row in existing_results}
    rows = read_jsonl(args.subset)
    if args.trajectory_id:
        selected_ids = set(args.trajectory_id)
        rows = [row for row in rows if row["trajectory_id"] in selected_ids]
    if args.limit is not None:
        rows = rows[: args.limit]
    samples = {row["task_id"]: row for row in json.loads(args.dataset.read_text(encoding="utf-8"))}
    if args.verify_reference_only:
        checks = []
        for row in rows:
            _, evaluator, score = extract_and_evaluate(
                row["task"], row["reference_output"], samples[row["source_instance_id"]]
            )
            checks.append({
                "reproduction_id": row["reproduction_id"],
                "task": row["task"],
                "reference_score": row["reference_score"],
                "rescored_reference": score,
                "match": abs(score - float(row["reference_score"])) <= 1e-12,
                "evaluator_output": evaluator,
            })
        output = {
            "checked": len(checks),
            "matched": sum(row["match"] for row in checks),
            "failed_ids": [row["reproduction_id"] for row in checks if not row["match"]],
        }
        (args.output_dir / "reference_evaluator_check.json").write_text(
            json.dumps({**output, "checks": checks}, indent=2), encoding="utf-8"
        )
        print(json.dumps(output))
        if output["matched"] != output["checked"]:
            raise SystemExit(1)
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = LocalQwen(args.model, args.revision, args.thinking_mode, args.max_new_tokens)

    run_config = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_argument": args.model,
        "revision_argument": args.revision,
        "resolved_model_commit": getattr(engine.model.config, "_commit_hash", None),
        "tokenizer_commit": engine.tokenizer.init_kwargs.get("_commit_hash"),
        "thinking_mode": args.thinking_mode,
        "temperature": 0.0,
        "do_sample": False,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    baseline_scores: dict[str, float] = {
        row["trajectory_id"]: float(row["reproduced_score"])
        for row in existing_results
        if row.get("status") == "completed"
        and row.get("condition") == "sharded_replay"
        and row.get("reproduced_score") is not None
    }
    for row in rows:
        if row["reproduction_id"] in completed:
            continue
        try:
            history, calls = replay(row, engine)
            output = history[-1]["content"]
            sample = samples[row["source_instance_id"]]
            answer, evaluator, score = extract_and_evaluate(row["task"], output, sample)
            if row["condition"] == "sharded_replay":
                baseline_scores[row["trajectory_id"]] = score
                reproduced_delta = None
                reproduced_effect = None
                effect_match = None
            else:
                baseline = baseline_scores.get(row["trajectory_id"])
                reproduced_delta = None if baseline is None else score - baseline
                reproduced_effect = None if reproduced_delta is None else effect_class(reproduced_delta)
                effect_match = None if reproduced_effect is None else reproduced_effect == row["reference_effect_class"]
            binary_outcome_match = None
            if row["task"] in BINARY_TASKS:
                binary_outcome_match = (score == 1.0) == (float(row["reference_score"]) == 1.0)
            record = {
                "reproduction_id": row["reproduction_id"],
                "trajectory_id": row["trajectory_id"],
                "target_turn": row.get("target_turn"),
                "task": row["task"],
                "condition": row["condition"],
                "reference_score": row["reference_score"],
                "reproduced_score": score,
                "reference_output": row["reference_output"],
                "reproduced_output": output,
                "score_match": abs(score - float(row["reference_score"])) <= 1e-12,
                "binary_outcome_match": binary_outcome_match,
                "reference_effect_class": row.get("reference_effect_class"),
                "reproduced_effect_class": reproduced_effect,
                "effect_class_match": effect_match,
                "reference_delta": row.get("reference_delta"),
                "reproduced_delta": reproduced_delta,
                "exact_output_match": output == row["reference_output"],
                "evaluator_output": evaluator,
                "extracted_answer": answer,
                "generated_history": history,
                "generation_calls": calls,
                "status": "completed",
            }
        except Exception as error:
            record = {
                "reproduction_id": row["reproduction_id"],
                "trajectory_id": row["trajectory_id"],
                "target_turn": row.get("target_turn"),
                "task": row["task"],
                "condition": row["condition"],
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        append_jsonl(result_path, record)
        print(json.dumps({"reproduction_id": row["reproduction_id"], "status": record["status"]}), flush=True)

    result_rows = read_jsonl(result_path)
    fields = [
        "reproduction_id", "trajectory_id", "target_turn", "task", "condition",
        "reference_score", "reproduced_score", "score_match", "binary_outcome_match",
        "reference_effect_class", "reproduced_effect_class", "effect_class_match",
        "reference_delta", "reproduced_delta", "exact_output_match", "status",
        "error_type", "error",
    ]
    with (args.output_dir / "reproduction_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in result_rows])


if __name__ == "__main__":
    main()

