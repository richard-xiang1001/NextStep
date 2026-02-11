#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_CONFIG="${ENV_CONFIG:-configs/env/webshop_v0.1.0.json}"
TASK_MANIFEST="${TASK_MANIFEST:-configs/env/webshop_task_manifest_v0.1.0.json}"
SPLIT="${SPLIT:-S_policy_dev}"
MAX_TASKS="${MAX_TASKS:-8}"
SEEDS="${SEEDS:-42}"
METHODS="${METHODS:-greedy}"
OUT_ROOT="${OUT_ROOT:-results/dev_real/$(date +%Y%m%d_%H%M%S)}"

LLM_PROVIDER="${LLM_PROVIDER:-openai}"
LLM_MODEL="${LLM_MODEL:-gpt-4.1-mini}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_STEPS="${MAX_STEPS:-80}"
MAX_TOK="${MAX_TOK:-6000}"
MAX_CALLS="${MAX_CALLS:-12}"
LLM_TOP_P="${LLM_TOP_P:-}"
LLM_STOP="${LLM_STOP:-}"
LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-}"
LLM_TOOLS_ENABLED="${LLM_TOOLS_ENABLED:-0}"

RM_CHECKPOINT="${RM_CHECKPOINT:-}"
RM_BON_NUM_SAMPLES="${RM_BON_NUM_SAMPLES:-4}"

PYTHON_BIN="python3"
if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
elif ! command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ "${LLM_PROVIDER}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[real-dev] OPENAI_API_KEY is required when LLM_PROVIDER=openai." >&2
  exit 1
fi
if [[ "${LLM_PROVIDER}" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "[real-dev] ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "[real-dev] ROOT=${ROOT_DIR}"
echo "[real-dev] PYTHON=${PYTHON_BIN}"
echo "[real-dev] OUT=${OUT_ROOT}"
echo "[real-dev] SPLIT=${SPLIT} MAX_TASKS=${MAX_TASKS}"
echo "[real-dev] METHODS=${METHODS}"
echo "[real-dev] SEEDS=${SEEDS}"
echo "[real-dev] LLM=${LLM_PROVIDER}/${LLM_MODEL}"
echo "[real-dev] BUDGET steps=${MAX_STEPS} tok=${MAX_TOK} calls=${MAX_CALLS}"
echo

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    run_dir="${OUT_ROOT}/${method}_seed${seed}"
    mkdir -p "${run_dir}"

    cmd=(
      "${PYTHON_BIN}" -m src.eval.baselines
      --method "${method}"
      --env_config "${ENV_CONFIG}"
      --task_manifest "${TASK_MANIFEST}"
      --split "${SPLIT}"
      --seed "${seed}"
      --llm_provider "${LLM_PROVIDER}"
      --llm_model "${LLM_MODEL}"
      --temperature "${TEMPERATURE}"
      --max_steps "${MAX_STEPS}"
      --max_tokens "${MAX_TOK}"
      --max_calls "${MAX_CALLS}"
      --max_tasks "${MAX_TASKS}"
      --output_dir "${run_dir}"
    )

    if [[ -n "${LLM_TOP_P}" ]]; then
      cmd+=(--llm_top_p "${LLM_TOP_P}")
    fi
    if [[ -n "${LLM_STOP}" ]]; then
      cmd+=(--llm_stop "${LLM_STOP}")
    fi
    if [[ -n "${LLM_REASONING_EFFORT}" ]]; then
      cmd+=(--llm_reasoning_effort "${LLM_REASONING_EFFORT}")
    fi
    if [[ "${LLM_TOOLS_ENABLED}" == "1" ]]; then
      cmd+=(--llm_tools_enabled)
    fi

    if [[ "${method}" == "rm_bon" ]]; then
      if [[ -z "${RM_CHECKPOINT}" ]]; then
        echo "[real-dev] RM_CHECKPOINT is required for method=rm_bon." >&2
        exit 1
      fi
      cmd+=(--num_samples "${RM_BON_NUM_SAMPLES}" --rm_checkpoint "${RM_CHECKPOINT}")
    fi

    if [[ "${method}" == "rm_mcts" ]]; then
      if [[ -z "${RM_CHECKPOINT}" ]]; then
        echo "[real-dev] RM_CHECKPOINT is required for method=rm_mcts." >&2
        exit 1
      fi
      cmd+=(--rm_checkpoint "${RM_CHECKPOINT}")
    fi

    echo "[real-dev] Running method=${method} seed=${seed}"
    "${cmd[@]}" 2>&1 | tee "${run_dir}/log.txt"
  done
done

summary_path="${OUT_ROOT}/summary.json"

"${PYTHON_BIN}" - "${OUT_ROOT}" "${summary_path}" <<'PY'
import json
import os
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []

for metrics_file in sorted(out_root.glob("*/metrics.json")):
    run_dir = metrics_file.parent
    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    except Exception:
        continue
    rows.append(
        {
            "run_dir": str(run_dir),
            "method": metrics.get("method"),
            "split": metrics.get("split"),
            "num_trajectories": metrics.get("num_trajectories"),
            "success_rate": metrics.get("success_rate"),
            "avg_tokens": metrics.get("avg_tokens"),
            "avg_llm_calls": metrics.get("avg_llm_calls"),
            "avg_step_count": metrics.get("avg_step_count"),
            "avg_env_step": metrics.get("avg_env_step"),
            "avg_policy_step": metrics.get("avg_policy_step"),
            "parse_fail_rate": metrics.get("parse_fail_rate"),
            "fallback_rate": metrics.get("fallback_rate"),
            "repair_trigger_rate": metrics.get("repair_trigger_rate"),
            "budget_cut_rate": metrics.get("budget_cut_rate"),
        }
    )

with open(summary_path, "w") as f:
    json.dump(rows, f, indent=2)

if not rows:
    print("[real-dev] No metrics found under", out_root)
    raise SystemExit(0)

print("\n[real-dev] Summary")
print("method\tseed_hint\tsuccess\tavg_tok\tavg_call\tavg_step\tavg_env_step")
for row in rows:
    run_name = Path(row["run_dir"]).name
    seed_hint = run_name.split("_seed")[-1] if "_seed" in run_name else "na"
    print(
        f"{row.get('method')}\t{seed_hint}\t{row.get('success_rate'):.4f}\t"
        f"{row.get('avg_tokens'):.1f}\t{row.get('avg_llm_calls'):.2f}\t"
        f"{row.get('avg_step_count'):.2f}\t{row.get('avg_env_step'):.2f}"
    )
print("\n[real-dev] Saved:", summary_path)
PY

