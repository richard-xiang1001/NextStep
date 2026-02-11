#!/usr/bin/env bash
set -euo pipefail

# -------- Config (override via env) --------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_CONFIG="${ENV_CONFIG:-configs/env/webshop_v0.1.0.json}"
TASK_MANIFEST="${TASK_MANIFEST:-configs/env/webshop_task_manifest_v0.1.0.json}"
SPLIT="${SPLIT:-S_smoke_1}"
SEED="${SEED:-42}"

# Baseline budgets (keep small for smoke)
MAX_STEPS="${MAX_STEPS:-20}"
MAX_TOK="${MAX_TOK:-800}"
MAX_CALLS="${MAX_CALLS:-6}"

# Fake LLM behavior
export NEXTSTEP_FAKE_INVALID_RATE="${NEXTSTEP_FAKE_INVALID_RATE:-0.25}"

# Output
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-results/smoke_fake/${TS}}"

PYTHON_BIN="python3"
if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
elif ! command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

# -------- Helpers --------
need_file() {
  if [[ ! -f "$1" ]]; then
    echo "[smoke] Missing file: $1" >&2
    exit 1
  fi
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[smoke] Missing command: $1" >&2
    exit 1
  fi
}

ensure_split_exists() {
  local manifest_abs="$1"
  local split_name="$2"
  local fallback_split="$3"
  local generated_manifest="$4"

  if "${PYTHON_BIN}" - "$manifest_abs" "$split_name" <<'PY'
import json
import sys
manifest_path, split_name = sys.argv[1], sys.argv[2]
with open(manifest_path, "r") as f:
    m = json.load(f)
ok = split_name in m.get("splits", {}) and len(m["splits"][split_name].get("task_ids", [])) > 0
sys.exit(0 if ok else 1)
PY
  then
    echo "$manifest_abs"
    return 0
  fi

  "${PYTHON_BIN}" - "$manifest_abs" "$split_name" "$fallback_split" "$generated_manifest" <<'PY'
import copy
import json
import sys
from pathlib import Path

src_manifest, split_name, fallback_split, out_manifest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src_manifest, "r") as f:
    data = json.load(f)

splits = data.get("splits", {})
if fallback_split not in splits or not splits[fallback_split].get("task_ids"):
    # Pick the first non-empty split as fallback.
    fallback_split = None
    for name, info in splits.items():
        if info.get("task_ids"):
            fallback_split = name
            break
if fallback_split is None:
    raise SystemExit("No non-empty split found in manifest.")

smoke_task_id = splits[fallback_split]["task_ids"][0]

new_data = copy.deepcopy(data)
new_data.setdefault("splits", {})
new_data["splits"][split_name] = {
    "purpose": "Smoke test single-task split (auto-generated)",
    "size": 1,
    "task_ids": [smoke_task_id],
    "leakage_control": "Smoke-only split for local/CI sanity checks",
}
new_data["total_tasks"] = int(new_data.get("total_tasks", 0))

out_path = Path(out_manifest)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(new_data, f, indent=2)
print(str(out_path))
PY
}

run_one() {
  local method="$1"
  shift
  local -a extra_args=("$@")
  local -a cmd=(
    "${PYTHON_BIN}" -m src.eval.baselines
    --method "${method}"
    --env_config "${ENV_CONFIG}"
    --task_manifest "${TASK_MANIFEST_RESOLVED}"
    --split "${SPLIT}"
    --seed "${SEED}"
    --llm_provider fake
    --llm_model fake
    --max_steps "${MAX_STEPS}"
    --max_tokens "${MAX_TOK}"
    --max_calls "${MAX_CALLS}"
    --output_dir "${OUT_ROOT}/${method}"
  )
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi

  echo "[smoke] Running method=${method} split=${SPLIT} out=${OUT_ROOT}/${method}"
  "${cmd[@]}"

  test -f "${OUT_ROOT}/${method}/metrics.json" || (echo "[smoke] metrics.json missing" && exit 1)
  test -f "${OUT_ROOT}/${method}/trajectories.jsonl" || (echo "[smoke] trajectories.jsonl missing" && exit 1)
  echo "[smoke] OK: ${method}"
}

# -------- Main --------
need_file "${ROOT_DIR}/${ENV_CONFIG}"
need_file "${ROOT_DIR}/${TASK_MANIFEST}"
need_cmd "${PYTHON_BIN}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_ROOT}"

TASK_MANIFEST_ABS="${ROOT_DIR}/${TASK_MANIFEST}"
TASK_MANIFEST_RESOLVED="${TASK_MANIFEST_ABS}"
TMP_MANIFEST="${OUT_ROOT}/_smoke_manifest.json"
TASK_MANIFEST_RESOLVED="$(ensure_split_exists "${TASK_MANIFEST_ABS}" "${SPLIT}" "S_final_test" "${TMP_MANIFEST}")"

echo "[smoke] ROOT=${ROOT_DIR}"
echo "[smoke] PYTHON=${PYTHON_BIN}"
echo "[smoke] MANIFEST=${TASK_MANIFEST_RESOLVED}"
echo "[smoke] SPLIT=${SPLIT}"
echo "[smoke] OUT=${OUT_ROOT}"
echo "[smoke] NEXTSTEP_FAKE_INVALID_RATE=${NEXTSTEP_FAKE_INVALID_RATE}"
echo

run_one "greedy"
run_one "rm_bon" "--num_samples" "4" "--rm_checkpoint" "fake_ckpt"
run_one "rm_mcts" "--rm_checkpoint" "fake_ckpt"

# Hard checks to catch schema regressions.
grep -q '"bon_candidates"' "${OUT_ROOT}/rm_bon/trajectories.jsonl" || (echo "[smoke] bon_candidates missing" && exit 1)
grep -q '"picked_index"' "${OUT_ROOT}/rm_bon/trajectories.jsonl" || (echo "[smoke] picked_index missing" && exit 1)
grep -q '"mcts_planning_' "${OUT_ROOT}/rm_mcts/trajectories.jsonl" || (echo "[smoke] mcts planning fields missing" && exit 1)

echo
echo "[smoke] All fake-baseline smoke tests passed."
echo "[smoke] Artifacts saved under: ${OUT_ROOT}"
