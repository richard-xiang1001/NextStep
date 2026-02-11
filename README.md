# NextStep: Learning to Orchestrate Multi-Agent Workflows

**Version**: v0.1.0
**Target Venue**: AAAI 2027
**Status**: Active Development (Week 1/12)

---

## Overview

NextStep is a research project that trains an orchestrator policy to generate multi-agent workflows using learned reward models (RM) on **unverifiable or weakly-verifiable interactive tasks**.

### Key Features

- ✅ **RM-guided orchestration**: Replaces hard-coded rewards with automatic reward modeling
- ✅ **Anti-hacking mechanisms**: RM ensemble + hard negative mining + rule anchors
- ✅ **Cost-aware optimization**: Explicit cost regularization for better Pareto efficiency
- ✅ **Robust to worker pool shift**: Trains with randomized worker subsets
- ✅ **Reproducible**: Frozen configs, versioned data, comprehensive logging

---

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### WebShop Setup (Submodule)

WebShop is integrated as a git submodule under `third_party/WebShop` (not a pip package in this repo workflow).

```bash
# Verify required WebShop data files (small split + human instructions)
ls third_party/WebShop/data/items_shuffle_1000.json \
   third_party/WebShop/data/items_ins_v2_1000.json \
   third_party/WebShop/data/items_human_ins.json

# Smoke test environment wrapper
python - <<'PY'
from src.env.webshop import create_webshop_env
from src.env.base import Action

env = create_webshop_env({"max_steps": 80, "num_products": 1000, "observation_mode": "text"})
obs, _ = env.reset(task_id="0")
step = env.step(Action(action_type="search", args={"query": "water bottle"}))
print("reset_obs_len=", len(obs.text), "step_reward=", step.reward, "done=", step.done)
PY
```

### Project Structure

```
NextStep/
├── configs/               # Configuration files (frozen for v0.1.0)
│   ├── env/              # Environment and task manifests
│   ├── rm/               # Reward model configs
│   └── orchestrator/     # Orchestrator configs
├── data/                 # Dataset storage
│   └── rm/
│       ├── raw/          # Raw trajectories
│       └── processed/    # Processed training data
├── src/                  # Source code
│   ├── env/              # Environment wrappers
│   ├── rm/               # Reward model code
│   ├── orchestrator/     # Orchestrator implementation
│   ├── eval/             # Evaluation scripts
│   └── utils/            # Utilities
├── scripts/              # Experiment scripts
│   ├── run_baselines.sh
│   ├── train_rm.sh
│   ├── train_orchestrator.sh
│   └── eval_all.sh
├── results/              # Experiment results
│   ├── exp_logs/         # Per-experiment logs
│   ├── metrics/          # Computed metrics
│   └── plots/            # Generated plots
├── artifacts/            # Model checkpoints
│   ├── models/
│   └── snapshots/
└── docs/                 # Documentation
    ├── research_spec.md  # Research specification (frozen)
    ├── notes.md          # Research notes
    ├── failure_cases.md  # Failure analysis
    └── patches/          # Version patches
```

### Running Experiments

#### 1. Run Baselines

```bash
./scripts/run_baselines.sh --method greedy --seed 42
./scripts/run_baselines.sh --method rm_bon --seed 42
./scripts/run_baselines.sh --method rm_mcts --seed 42
```

#### 2. Train Reward Model

```bash
./scripts/train_rm.sh --config configs/rm/rm_training_v0.1.0.json
```

#### 3. Train Orchestrator

```bash
./scripts/train_orchestrator.sh \
  --rm-checkpoint artifacts/models/rm_ensemble \
  --max-iterations 100
```

#### 4. Evaluate and Generate Results

```bash
./scripts/eval_all.sh \
  --checkpoint artifacts/models/orchestrator_final \
  --split S_final_test
```

### CI + Smoke

Use fake-provider smoke tests for fast regression checks (no API key required):

```bash
./scripts/smoke_baselines_fake.sh
```

Optional overrides:

```bash
NEXTSTEP_FAKE_INVALID_RATE=0.25 \
SPLIT=S_smoke_1 \
SEED=42 \
MAX_STEPS=20 \
MAX_TOK=800 \
MAX_CALLS=6 \
OUT_ROOT=results/smoke_fake/local \
./scripts/smoke_baselines_fake.sh
```

GitHub Actions workflow:

- `.github/workflows/smoke-fake-baselines.yml`
- Matrix checks:
  - `smoke (0.0)` (normal path)
  - `smoke (0.25)` (repair path coverage)

Recommended branch protection required checks:

- `smoke (0.0)`
- `smoke (0.25)`

### Real-LLM Dev Slice (L0 -> L1)

Run a small real-provider slice first (defaults: `S_policy_dev`, `8` tasks, `greedy`):

```bash
export OPENAI_API_KEY=...
chmod +x scripts/run_real_dev_slice.sh
scripts/run_real_dev_slice.sh
```

Example with explicit methods and task count:

```bash
OPENAI_API_KEY=... \
METHODS="greedy rm_bon rm_mcts" \
MAX_TASKS=20 \
RM_CHECKPOINT=artifacts/models/rm_ensemble \
scripts/run_real_dev_slice.sh
```

Outputs:
- per-run artifacts under `results/dev_real/<timestamp>/<method>_seed<seed>/`
- aggregate summary at `results/dev_real/<timestamp>/summary.json`

---

## Research Questions

1. **RQ1 (Efficacy)**: Can RM-guided orchestration improve task success on unverifiable tasks under fixed budget?
2. **RQ2 (Efficiency)**: Can it improve success-cost Pareto?
3. **RQ3 (Robustness)**: Can it generalize when worker pool changes?
4. **RQ4 (Anti-hacking)**: Can we reduce RM gaming while maintaining performance?

---

## Key Components

### Orchestrator
- **Model**: Trainable policy `O_θ` that outputs workflow JSON
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Output**: `[{subtask, worker_id, access_list}, ...]`

### Reward Model
- **Input**: Query + trajectory summary
- **Output**: Utility score in [0, 1]
- **Training**: Offline pairwise ranking + online hard-negative mining
- **Anti-hacking**: Ensemble of 3 RMs (min aggregation)

### Worker Pool
- **Size**: 4 workers (fixed for v0.1.0)
- **Training**: Randomly sample 3/4 per episode
- **Roles**: general_reasoning, tool_use, planning, verification

---

## Budget Limits (v0.1.0)

| Resource | Limit |
|----------|-------|
| Max workflow steps (`T_max`) | 6 |
| Max env steps (`B_step`) | 80 |
| Max tokens (`B_tok`) | 6000 |
| Max worker calls (`B_call`) | 12 |
| Max tokens per call | 512 |
| Rollout count (`K`) | 16 (main), 32 (ablation) |

---

## Documentation

- **Research Spec**: See `docs/research_spec.md` for complete research design
- **Notes**: See `docs/notes.md` for ongoing research notes
- **Patches**: See `docs/patches/` for version history and changes

---

## Citation

```bibtex
@article{nextstep2027,
  title={Learning to Orchestrate Multi-Agent Workflows with Reward Models for Unverifiable Tasks},
  author={Author Name},
  journal={AAAI},
  year={2027}
}
```

---

## License

[To be determined]

---

## Acknowledgments

This work is inspired by:
- **ARMAP**: Automatic Reward Modeling and Planning
- **GPTSwarm**: Language Agents as Optimizable Graphs
- **The Conductor**: Learning to Orchestrate Agents in Natural Language
