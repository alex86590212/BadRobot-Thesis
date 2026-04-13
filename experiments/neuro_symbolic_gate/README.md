# Neuro-symbolic action gate experiment

Post-planning symbolic gate for the BadRobot **digital** embodied agent: the LLM still uses [`digital_evaluation/system_prompt.py`](../../digital_evaluation/system_prompt.py); after each response, `nsg` parses the `function` list and checks it against `rules/rsafety_v0.yaml` (`R_safety` v0).

## Setup

From **BadRobot-Thesis** (repository root):

```bash
pip install -r experiments/neuro_symbolic_gate/requirements.txt
export OPENAI_API_KEY=sk-...
```

## Run gated sweep (primary)

```bash
cd BadRobot-Thesis
python3 experiments/neuro_symbolic_gate/scripts/run_experiment.py \
  --model gpt-4-turbo \
  --attack_method "safety misalignment" \
  --split malicious \
  --limit 20 \
  --out experiments/neuro_symbolic_gate/outputs/gated_sample.jsonl
```

- `--split safe` uses [`data/safe_control_instructions.jsonl`](data/safe_control_instructions.jsonl) (false-positive check).
- `--split both` runs malicious then safe into the same `--out` file.

## Metrics vs baseline `.txt`

```bash
python3 experiments/neuro_symbolic_gate/scripts/compare_metrics.py \
  --gated experiments/neuro_symbolic_gate/outputs/gated_sample.jsonl \
  --baseline_txt "gpt-4-turbo_safety misalignment_results.txt"
```

Omit `--baseline_txt` if you only have gated JSONL. Figures go to `outputs/figures/` unless `--no_plots`.

## Optional: baseline driver + JSONL

From `digital_evaluation/` (so `system_prompt` imports resolve):

```bash
cd BadRobot-Thesis/digital_evaluation
python attack_main.py --api_key "$OPENAI_API_KEY" --base_url https://api.openai.com/v1 \
  --model gpt-4-turbo --attack_method "safety misalignment" --load_malicious_queries True \
  --nsg_gate_jsonl ../experiments/neuro_symbolic_gate/outputs/from_attack_main.jsonl
```

## Layout

| Path | Role |
|------|------|
| `nsg/plan_parser.py` | Parse model `function` entries |
| `nsg/rule_engine.py` | Load YAML and match violations |
| `nsg/safety_gate.py` | `evaluate_plan` / `evaluate_raw_model_response` |
| `nsg/runner_core.py` | OpenAI loop + attack composition + JSONL row schema |
| `rules/rsafety_v0.yaml` | Rule IDs and patterns |
| `data/safe_control_instructions.jsonl` | Benign control queries |

## ASP extension (optional)

To add `clingo`, implement `nsg/asp_backend.py` and point the gate at `.lp` facts; keep the same JSONL schema. Not included in the default path.
