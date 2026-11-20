from openai import OpenAI, AzureOpenAI
import os, time, json, re, hashlib
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv(Path(__file__).resolve().parent / ".env")

def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages

class OpenAI_Model:
    def __init__(self):
        self.client = None
        self.client_signature = None

    def _current_signature(self):
        use_azure = "AZURE_OPENAI_API_KEY" in os.environ and "AZURE_OPENAI_ENDPOINT" in os.environ
        if use_azure:
            return (
                "azure",
                os.environ.get("AZURE_OPENAI_API_KEY", ""),
                os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            )
        return (
            "openai",
            os.environ.get("OPENAI_API_KEY", ""),
            os.environ.get("OPENAI_BASE_URL", ""),
        )

    def _ensure_client(self):
        signature = self._current_signature()
        if self.client is not None and signature == self.client_signature:
            return
        if signature[0] == "azure":
            self.client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2024-10-01-preview",
            )
        else:
            assert "OPENAI_API_KEY" in os.environ
            client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
            if os.environ.get("OPENAI_BASE_URL"):
                client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
            self.client = OpenAI(**client_kwargs)
        self.client_signature = signature

    def cost_calculator(self, model, usage, is_batch_model=False):
        is_finetuned, base_model = False, model
        if model.startswith("ft:gpt"):
            is_finetuned = True
            base_model = model.split(":")[1]

        prompt_tokens = usage['prompt_tokens']
        if 'prompt_tokens_details' in usage and usage['prompt_tokens_details']:
            prompt_tokens_cached = usage['prompt_tokens_details'].get('cached_tokens', 0)
        else:
            prompt_tokens_cached = 0
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached

        completion_tokens = usage['completion_tokens']
        if base_model.startswith("gpt-4o-mini"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.0003, 0.00015
            else:
                inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.00375, 0.015
            else:
                inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model == "gpt-5.6-luna":
            inp_token_cost, out_token_cost = 0.001, 0.006
        elif base_model == "deepseek-ai/DeepSeek-V3.2":
            inp_token_cost, out_token_cost = 0.002, 0.003
        elif base_model == "Qwen/Qwen3-14B":
            inp_token_cost, out_token_cost = 0.002, 0.020
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif base_model.startswith("o1-mini"):
            inp_token_cost, out_token_cost = 0.003, 0.012
        elif base_model.startswith("gpt-4.5-preview"):
            inp_token_cost, out_token_cost = 0.075, 0.150
        elif base_model.startswith("o1-preview") or base_model == "o1":
            inp_token_cost, out_token_cost = 0.015, 0.06
        else:
            print(f"[pricing] Model {model} pricing unknown; recording total_usd as null")
            return None

        cache_discount = 0.5 # cached tokens are half the price
        batch_discount = 0.5 # batch API is half the price
        total_usd = ((prompt_tokens_non_cached + prompt_tokens_cached * cache_discount) / 1000) * inp_token_cost + (completion_tokens / 1000) * out_token_cost
        if is_batch_model:
            total_usd *= batch_discount

        return total_usd

    def generate(self, messages, model="gpt-4o-mini", timeout=30, max_retries=3, temperature=1.0, is_json=False, return_metadata=False, max_tokens=None, variables={}, call_role=None):
        kwargs = {}
        if is_json:
            kwargs["response_format"] = { "type": "json_object" }
        N = 0
        self._ensure_client()
        if os.environ.get("P1_CANONICAL_LOG"):
            max_retries = min(max_retries, int(os.environ.get("P1_MAX_ATTEMPTS", "3")))

        messages = format_messages(messages, variables)
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()

        # o1- models do not support system message. If the first message is a system message, and the second message is a user message, then prepend the user message with the system message.
        if model.startswith("o1") and len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
            system_message = messages[0]["content"]
            messages[1]["content"] = f"System Message: {system_message}\n{messages[1]['content']}"
            messages = messages[1:]

        last_error_summary = ""
        while True:
            stop_file = os.environ.get("P1_BUDGET_STOP_FILE")
            if stop_file and Path(stop_file).exists():
                raise RuntimeError("P1 budget monitor requested a stop before API call")
            attempt_number = N + 1
            attempt_id = hashlib.sha256(
                f"{started_at.isoformat()}|{model}|{attempt_number}|{call_role}".encode(
                    "utf-8"
                )
            ).hexdigest()[:24]
            response = None
            try:
                response = self.client.chat.completions.create(model=model, messages=messages, timeout=timeout, max_completion_tokens=max_tokens, temperature=temperature, **kwargs)
                response = response.to_dict()
                message = (response.get("choices") or [{}])[0].get("message") or {}
                response_text = message.get("content")
                if not isinstance(response_text, str):
                    raise ValueError(
                        "Successful proxy response lacks textual message.content"
                    )
                break
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                err_cls = type(e).__name__
                err_msg = str(e)
                last_error_summary = f"{err_cls} | status_code={status_code} | message={err_msg[:400]}"
                N += 1
                final_failure = N >= max_retries
                self._log_p1_call({
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "started_at": started_at.isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "latency_seconds": time.perf_counter() - started_clock,
                    "requested_model": model,
                    "returned_model": None,
                    "response_id": None,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "retry_count": max(0, N - 1),
                    "status": "unrecovered_error" if final_failure else "retry_error",
                    "error_type": err_cls,
                    "error_summary": last_error_summary,
                    "raw_messages": messages,
                    "raw_response": response if isinstance(response, dict) else None,
                    "usage": None,
                }, call_role=call_role)
                if N >= max_retries:
                    if os.environ.get("MODEL_OPENAI_DEBUG_ERRORS", "").lower() in {"1", "true", "yes"}:
                        raise Exception(f"Failed to get response from OpenAI | underlying={last_error_summary}")
                    raise Exception("Failed to get response from OpenAI")
                else:
                    time.sleep(min(16, 4 * (2 ** (N - 1))))

        usage = response['usage']
        total_usd = self.cost_calculator(model, usage)
        prompt_tokens_cached = 0
        if 'prompt_tokens_details' in usage and usage['prompt_tokens_details']:
            prompt_tokens_cached = usage['prompt_tokens_details'].get('cached_tokens', 0)
        call_metadata = {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": time.perf_counter() - started_clock,
            "requested_model": model,
            "returned_model": response.get("model"),
            "response_id": response.get("id"),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "retry_count": N,
            "status": "success",
            "error_type": None,
            "raw_messages": messages,
            "raw_response": response_text,
            "usage": usage,
        }
        self._log_p1_call(call_metadata, call_role=call_role)

        if not return_metadata:
            return response_text
        return {"message": response_text, "total_tokens": usage['total_tokens'], "prompt_tokens": usage['prompt_tokens'], "prompt_tokens_cached": prompt_tokens_cached, "completion_tokens": usage['completion_tokens'], "total_usd": total_usd, "requested_model": model, "returned_model": response.get("model"), "response_id": response.get("id"), "usage": usage, "retry_count": N}

    def _log_p1_call(self, record, call_role=None):
        log_path = os.environ.get("P1_CANONICAL_LOG")
        if not log_path:
            return
        context = json.loads(os.environ.get("P1_RUN_CONTEXT", "{}"))
        record.update(context)
        if call_role is not None:
            record["call_role"] = call_role
        record["provider_alias"] = os.environ.get("P1_PROVIDER_ALIAS", "api2_proxy")
        record["base_url_alias"] = "OPENAI_BASE_URL"
        record["endpoint_type"] = "openai_chat_completions"
        record["prompt_hash"] = hashlib.sha256(
            json.dumps(record.get("raw_messages"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _extract_balanced_json(self, text):
        stack = []
        start_idx = None
        in_string = False
        escape = False
        opening = {"{": "}", "[": "]"}
        closing = {"}": "{", "]": "["}

        for idx, char in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char in opening:
                if start_idx is None:
                    start_idx = idx
                stack.append(char)
                continue

            if char in closing and stack:
                if stack[-1] == closing[char]:
                    stack.pop()
                    if not stack and start_idx is not None:
                        return text[start_idx:idx + 1]
        return None

    def _parse_json_response(self, raw_text, model):
        parse_errors = []
        candidates = [raw_text.strip()]

        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())

        balanced = self._extract_balanced_json(raw_text)
        if balanced:
            candidates.append(balanced.strip())

        for candidate in candidates:
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as error:
                parse_errors.append(str(error))

        excerpt = raw_text.strip().replace("\n", "\\n")[:240]
        raise ValueError(f"Failed to parse JSON response for model {model}. Excerpt: {excerpt}. Errors: {parse_errors}")

    def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
        response = self.generate(messages, model, is_json=True, **kwargs)
        response["message"] = json.loads(response["message"])
        return response


model = OpenAI_Model()
generate = model.generate
generate_json = model.generate_json





if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Humor is a way to make people laugh. Tell me a joke about UC Berkeley."},
        {"role": "assistant", "content": '{"joke": '}
    ]

    model = "gpt-4o"
    response = generate(messages, model=model, return_metadata=True)
    
    print(response)

    # batch_id = client.generate_w_delay(messages, model="gpt-4o-mini")
    # print(batch_id)
