from __future__ import annotations

import hashlib
import json
from typing import Callable

ACK = "Acknowledged."


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_messages(messages: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": row["role"], "content": row["content"]}
        for row in messages
        if row.get("role") in {"system", "user", "assistant"}
    ]


def split_turns(messages: list[dict]) -> tuple[dict, list[str], list[str]]:
    rows = semantic_messages(messages)
    systems = [row for row in rows if row["role"] == "system"]
    users = [row["content"] for row in rows if row["role"] == "user"]
    assistants = [row["content"] for row in rows if row["role"] == "assistant"]
    if len(systems) != 1:
        raise ValueError(f"Expected one system message, found {len(systems)}")
    if len(users) != len(assistants):
        raise ValueError("Original trace does not contain one assistant reply per user")
    return dict(systems[0]), users, assistants


def _generated_row(index: int, result: dict) -> dict:
    return {
        "turn_index": index,
        "assistant": result["message"],
        "requested_model": result.get("requested_model"),
        "returned_model": result.get("returned_model"),
        "response_id": result.get("response_id"),
    }


def replay_after_target(
    original_messages: list[dict],
    model: str,
    target_turn: int,
    replace_target: bool,
    call: Callable[[list[dict], str], dict],
) -> tuple[list[dict], list[dict]]:
    """Replay all assistants after target; preserve or neutralize A_t."""
    system, users, assistants = split_turns(original_messages)
    if target_turn < 0 or target_turn >= len(users) - 1:
        raise ValueError("Target must be a non-final assistant turn")
    history = [system]
    generated: list[dict] = []
    for index, user in enumerate(users):
        history.append({"role": "user", "content": user})
        if index < target_turn:
            history.append({"role": "assistant", "content": assistants[index]})
        elif index == target_turn:
            history.append({
                "role": "assistant",
                "content": ACK if replace_target else assistants[index],
            })
        else:
            result = call(history, model)
            history.append({"role": "assistant", "content": result["message"]})
            generated.append(_generated_row(index, result))
    validate_p0a_history(original_messages, history, target_turn, replace_target, generated)
    return history, generated


def final_only_history(
    original_messages: list[dict],
    model: str,
    condition: str,
    call: Callable[[list[dict], str], dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build RQ2 context and generate only the final assistant answer."""
    system, users, assistants = split_turns(original_messages)
    history = [system]
    slots: list[dict] = []
    for index, user in enumerate(users[:-1]):
        history.append({"role": "user", "content": user})
        if condition == "null_replay":
            text = assistants[index]
        elif condition == "short_neutralization":
            text = ACK
        elif condition == "length_matched_neutralization":
            count = max(1, len((assistants[index] or "").split()))
            text = " ".join([ACK] * count)
            slots.append({
                "turn_index": index,
                "original_estimated_tokens": len((assistants[index] or "").split()),
                "replacement_estimated_tokens": len(text.split()),
            })
        else:
            raise ValueError(f"Unknown P0-B condition: {condition}")
        history.append({"role": "assistant", "content": text})
    history.append({"role": "user", "content": users[-1]})
    result = call(history, model)
    history.append({"role": "assistant", "content": result["message"]})
    generated = [_generated_row(len(users) - 1, result)]
    validate_p0b_history(original_messages, history, condition)
    return history, generated, slots


def validate_p0a_history(
    original_messages: list[dict],
    replayed: list[dict],
    target_turn: int,
    replace_target: bool,
    generated: list[dict],
) -> None:
    _, old_users, old_assistants = split_turns(original_messages)
    _, new_users, new_assistants = split_turns(replayed)
    if new_users != old_users:
        raise ValueError("Frozen user sequence changed")
    if new_assistants[:target_turn] != old_assistants[:target_turn]:
        raise ValueError("Pre-target assistant prefix changed")
    expected_target = ACK if replace_target else old_assistants[target_turn]
    if new_assistants[target_turn] != expected_target:
        raise ValueError("Target assistant content violates condition")
    expected_positions = list(range(target_turn + 1, len(old_assistants)))
    if [row["turn_index"] for row in generated] != expected_positions:
        raise ValueError("Not every post-target assistant position was regenerated")


def validate_p0b_history(original_messages: list[dict], replayed: list[dict], condition: str) -> None:
    _, old_users, old_assistants = split_turns(original_messages)
    _, new_users, new_assistants = split_turns(replayed)
    if new_users != old_users:
        raise ValueError("Frozen user sequence changed")
    if len(new_assistants) != len(old_assistants):
        raise ValueError("RQ2 replay changed turn count")
    context = new_assistants[:-1]
    if condition == "null_replay" and context != old_assistants[:-1]:
        raise ValueError("Null-Replay changed assistant history")
    if condition == "short_neutralization" and any(text != ACK for text in context):
        raise ValueError("Short neutralization contains non-ACK history")
    if condition == "length_matched_neutralization":
        for old, new in zip(old_assistants[:-1], context):
            if len((old or "").split()) != len((new or "").split()):
                raise ValueError("Length-matched history violates frozen whitespace count")


def user_sequence_hash(messages: list[dict]) -> str:
    _, users, _ = split_turns(messages)
    return content_hash(users)


def exact_returned_model(generated: list[dict]) -> str | None:
    values = {row.get("returned_model") for row in generated if row.get("returned_model")}
    if len(values) > 1:
        raise ValueError(f"One condition crossed returned model IDs: {sorted(values)}")
    return next(iter(values)) if values else None

