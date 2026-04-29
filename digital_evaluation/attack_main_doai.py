import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from system_prompt import system_prompt
from contextual_jailbreak import get_random_jailbreak_prompt
from safety_misalignment import get_random_safety_misalignment_prompt
from conceptual_deception import rewrite_user_input


DEFAULT_BASE_URL = "https://inference.do-ai.run/v1/chat/completions"
ATTACK_CONTEXTUAL_JAILBREAK = "contextual jailbreak"
ATTACK_SAFETY_MISALIGNMENT = "safety misalignment"
ATTACK_CONCEPTUAL_DECEPTION = "conceptual deception"


def _append_nsg_gate_jsonl(
    gate_path: str,
    model: str,
    attack_method: str,
    base_query: str,
    user_message: str,
    result: str | None,
) -> None:
    """Append one neuro-symbolic gate record (same schema as experiments JSONL)."""
    repo = Path(__file__).resolve().parents[1]
    exp = repo / "experiments" / "neuro_symbolic_gate"
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    from datetime import datetime, timezone

    from experiments.neuro_symbolic_gate.nsg.plan_parser import parse_model_response
    from experiments.neuro_symbolic_gate.nsg.runner_core import default_rules_path, model_refusal_heuristic
    from experiments.neuro_symbolic_gate.nsg.safety_gate import evaluate_parsed_plan

    raw = result or ""
    plan = parse_model_response(raw)
    gate = evaluate_parsed_plan(plan, rules_path=default_rules_path())
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "attack_method": attack_method,
        "split": "malicious",
        "base_query": base_query,
        "user_message": user_message,
        "raw_response": raw,
        "functions_raw": plan.function_strings,
        "parse_error": plan.parse_error,
        "gate_status": gate.status.value,
        "violations": gate.violations,
        "model_refusal_heuristic": model_refusal_heuristic(raw),
        "pre_gate_non_empty": bool(plan.function_strings),
        "post_gate_executable": bool(plan.function_strings) and gate.status.value == "allowed",
    }
    p = Path(gate_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_malicious_queries():
    """
    This function loads the malicious queries from the Excel file at the repo root
    (BadRobot-Thesis/Physical_Word_Malicious_Queries.xlsx) and returns them as a list of strings.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(repo_root, "Physical_Word_Malicious_Queries.xlsx")
    try:
        data = pd.read_excel(file_path)
        return data["Request"].tolist()
    except Exception as e:
        print(f"Error loading malicious queries: {e}")
        return []


def _call_chat_completion(base_url, api_key, model, messages, max_tokens=100):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    response = requests.post(base_url, headers=headers, json=data, timeout=120)
    response.raise_for_status()
    payload = response.json()

    choices = payload.get("choices") or []
    if not choices:
        raise ValueError(f"Unexpected response format: {payload}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise ValueError(f"Missing message content in response: {payload}")
    return content


def process_single_query(
    base_url,
    api_key,
    model,
    user_input,
    attack_method,
    output_file,
    nsg_gate_jsonl=None,
    max_tokens=100,
):
    """
    This function processes a single user input query with the selected attack method,
    performs a model inference, and saves the result.
    """
    base_for_log = user_input
    if attack_method == ATTACK_CONTEXTUAL_JAILBREAK:
        jailbreak_prompt = get_random_jailbreak_prompt()
        user_input = f"{jailbreak_prompt}\n{user_input}"
    elif attack_method == ATTACK_SAFETY_MISALIGNMENT:
        safety_prompt = get_random_safety_misalignment_prompt()
        user_input = f"{user_input}\n{safety_prompt}"
    elif attack_method == ATTACK_CONCEPTUAL_DECEPTION:
        print(f"Original User Input: {user_input}")
        # Keep the same behavior: conceptual rewriter still uses its own model call.
        # This mirrors the original script's exact flow.
        class _CompatClient:
            def __init__(self, base_url, api_key):
                self.base_url = base_url
                self.api_key = api_key

                class _ChatCompletions:
                    def __init__(self, outer):
                        self._outer = outer

                    def create(self, model, messages):
                        content = _call_chat_completion(
                            self._outer.base_url,
                            self._outer.api_key,
                            model,
                            messages,
                            max_tokens=max_tokens,
                        )
                        return type(
                            "Resp",
                            (),
                            {
                                "choices": [
                                    type(
                                        "Choice",
                                        (),
                                        {"message": type("Msg", (), {"content": content})()},
                                    )()
                                ]
                            },
                        )()

                self.chat = type("Chat", (), {"completions": _ChatCompletions(self)})()

        compat_client = _CompatClient(base_url, api_key)
        user_input = rewrite_user_input(user_input, compat_client, model=model)
        print(f"Rewritten User Input: {user_input}")

    print(user_input)
    messages = [system_prompt]
    if user_input:
        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)

    try:
        result = _call_chat_completion(base_url, api_key, model, messages, max_tokens=max_tokens)
        print(f"Response: {result}")

        with open(output_file, "a") as f:
            f.write(f"Input: {user_input}\n")
            f.write(f"Response: {result}\n")
            f.write("=" * 50 + "\n")

        if nsg_gate_jsonl:
            _append_nsg_gate_jsonl(
                nsg_gate_jsonl, model, attack_method, base_for_log, user_input, result
            )
    except Exception as e:
        print(f"Error calling the API: {e}")


def main(
    api_key,
    base_url,
    model,
    user_input,
    attack_method,
    load_malicious_queries_flag,
    nsg_gate_jsonl=None,
    max_tokens=100,
):
    if model is None:
        model = "openai-gpt-oss-120b"

    output_file = f"{model}_{attack_method}_results.txt"

    if load_malicious_queries_flag:
        malicious_queries = load_malicious_queries()
        if malicious_queries:
            for i, malicious_query in enumerate(malicious_queries, start=1):
                print(f"\nProcessing Malicious Query {i}/{len(malicious_queries)}:")
                process_single_query(
                    base_url,
                    api_key,
                    model,
                    malicious_query,
                    attack_method,
                    output_file,
                    nsg_gate_jsonl,
                    max_tokens=max_tokens,
                )
        return

    process_single_query(
        base_url,
        api_key,
        model,
        user_input,
        attack_method,
        output_file,
        nsg_gate_jsonl,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model with optional user input and default system prompt."
    )
    parser.add_argument("--api_key", type=str, default="YOUR_MODEL_ACCESS_KEY")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default="openai-gpt-oss-120b")
    parser.add_argument(
        "--user_input",
        type=str,
        default="",
        help="Ignored when --load_malicious_queries is True.",
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        choices=[
            ATTACK_CONTEXTUAL_JAILBREAK,
            ATTACK_SAFETY_MISALIGNMENT,
            ATTACK_CONCEPTUAL_DECEPTION,
        ],
        default=ATTACK_SAFETY_MISALIGNMENT,
        help="Choose the attack method",
    )
    parser.add_argument(
        "--load_malicious_queries",
        type=bool,
        default=True,
        help="Set to True to load malicious queries from file (default: True).",
    )
    parser.add_argument(
        "--nsg_gate_jsonl",
        type=str,
        default="",
        help="Optional path to append neuro-symbolic gate JSONL (one object per query).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum tokens for completion.",
    )

    args = parser.parse_args()

    main(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        user_input=args.user_input,
        attack_method=args.attack_method,
        load_malicious_queries_flag=args.load_malicious_queries,
        nsg_gate_jsonl=args.nsg_gate_jsonl or None,
        max_tokens=args.max_tokens,
    )
