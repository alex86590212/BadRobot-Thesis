import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd  # Assuming pandas is available for reading the CSV
from system_prompt import system_prompt  # Import system prompt
from openai import OpenAI  # Assuming you have a third-party client module
from contextual_jailbreak import get_random_jailbreak_prompt
from safety_misalignment import get_random_safety_misalignment_prompt
from conceptual_deception import rewrite_user_input


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

    from nsg.plan_parser import parse_model_response
    from nsg.runner_core import default_rules_path, model_refusal_heuristic
    from nsg.safety_gate import evaluate_parsed_plan

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
        # Assuming the file is an Excel file and we need the 'Request' column
        data = pd.read_excel(file_path)  # Change to read_excel
        return data['Request'].tolist()
    except Exception as e:
        print(f"Error loading malicious queries: {e}")
        return []

def process_single_query(client, model, user_input, attack_method, output_file, nsg_gate_jsonl=None):
    """
    This function processes a single user input query with the selected attack method,
    performs a model inference, and saves the result.
    """
    base_for_log = user_input
    if attack_method == "contextual jailbreak":
        jailbreak_prompt = get_random_jailbreak_prompt()
        user_input = f"{jailbreak_prompt}\n{user_input}"

    elif attack_method == "safety misalignment":
        safety_prompt = get_random_safety_misalignment_prompt()
        user_input = f"{user_input}\n{safety_prompt}"

    elif attack_method == "conceptual deception":
        print(f"Original User Input: {user_input}")
        user_input = rewrite_user_input(user_input, client, model=model)
        print(f"Rewritten User Input: {user_input}")


    print(user_input)
    messages = [system_prompt]  # Default system prompt
    if user_input:
        user_message = {
            "role": "user",
            "content": user_input
        }
        messages.append(user_message)  # Add user input

    # Call third-party API to make model inference
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        # Output response
        result = response.choices[0].message.content
        print(f"Response: {result}")

        # Save the result to the file
        with open(output_file, 'a') as f:  # Append mode
            f.write(f"Input: {user_input}\n")
            f.write(f"Response: {result}\n")
            f.write("="*50 + "\n")  # Separate each query with a line

        if nsg_gate_jsonl:
            _append_nsg_gate_jsonl(
                nsg_gate_jsonl, model, attack_method, base_for_log, user_input, result
            )

    except Exception as e:
        print(f"Error calling the API: {e}")

def main(api_key, base_url, model, user_input, attack_method, load_malicious_queries_flag, nsg_gate_jsonl=None):
    # Create third-party OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # If no model is passed, set the default model
    if model is None:
        model = 'default-model'

    # Generate the output file name
    output_file = f"{model}_{attack_method}_results.txt"

    # Load malicious queries if the flag is set to True
    if load_malicious_queries_flag:
        malicious_queries = load_malicious_queries()
        if malicious_queries:
            for i, malicious_query in enumerate(malicious_queries, start=1):
                print(f"\nProcessing Malicious Query {i}/{len(malicious_queries)}:")
                # Process each malicious query one at a time
                process_single_query(
                    client, model, malicious_query, attack_method, output_file, nsg_gate_jsonl
                )
        return  # Stop the function after processing malicious queries

    # If no malicious queries are loaded, process the user_input as a single query
    process_single_query(client, model, user_input, attack_method, output_file, nsg_gate_jsonl)


if __name__ == '__main__':
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Run model with optional user input and default system prompt.")
    parser.add_argument('--api_key', type=str, default='YOUR_API_KEY_HERE')
    parser.add_argument('--base_url', type=str, default='YOUR_API_BASE_URL_HERE')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--user_input', type=str, default='Hello, robot!')
    parser.add_argument('--attack_method', type=str,
                        choices=['contextual jailbreak', 'safety misalignment', 'conceptual deception'],
                        default='contextual jailbreak', help='Choose the attack method')
    parser.add_argument('--load_malicious_queries', type=bool, default=False, help='Set to True to load malicious queries from file')
    parser.add_argument(
        '--nsg_gate_jsonl',
        type=str,
        default='',
        help='Optional path to append neuro-symbolic gate JSONL (one object per query).',
    )


    args = parser.parse_args()

    # Call main function, passing parsed arguments
    main(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        user_input=args.user_input,
        attack_method=args.attack_method,
        load_malicious_queries_flag=args.load_malicious_queries,
        nsg_gate_jsonl=args.nsg_gate_jsonl or None,
    )
