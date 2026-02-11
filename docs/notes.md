# Research Notes (NextStep v0.1.0)

## Project Overview

**Goal**: Train an orchestrator policy that generates multi-agent workflows using learned reward models on unverifiable interactive tasks.

**Target Venue**: AAAI 2027

**Timeline**: 12 weeks (Feb - May 2026)

---

## Key Design Decisions

### Week 1 (Feb 10-16, 2026)

- [x] Project structure created
- [x] Python environment setup (venv + dependencies)
- [x] WebShop integration (git submodule + wrapper)
- [x] Environment and baseline implementation
- [x] Task splits generated (1000 tasks, 5 splits)
- [x] WebShop data download and setup
- [x] Single-agent baselines hardened (budget envelope + repair + structured logs)
- [x] Offline fake-smoke baseline run (greedy/rm_bon/rm_mcts)
- [x] CI smoke workflow for baseline regression checks
- [ ] Run first real-LLM baseline experiment

### Decisions Log

#### 2026-02-10: Project Structure & Initial Implementation
- **Decision**: Created full directory structure following research_spec.md
- **Rationale**: Ensure all configs, data, code, logs are properly organized
- **Files Created**:
  - `configs/env/`: Environment and task manifest configs (✅ completed)
  - `configs/rm/`: Reward model training configs (✅ completed)
  - `configs/orchestrator/`: Orchestrator RL configs (✅ completed)
  - `src/`: Source code structure (✅ implemented core modules)
  - `scripts/`: Experiment scripts (✅ all 4 scripts created)
  - `docs/`: Documentation and notes (✅ templates created)

#### 2026-02-10: WebShop Integration Strategy
- **Decision**: Use WebShop as git submodule instead of pip package
- **Rationale**: WebShop doesn't provide standard pip package, requires manual setup
- **Implementation**:
  - Added `third_party/WebShop` as git submodule (commit: 64fa2a5)
  - Created `WebShopEnvironment` wrapper in `src/env/webshop.py`
  - Designed to work with both small (1000) and full (1.18M) product datasets

#### 2026-02-10: Task Split Generation
- **Decision**: Generate random splits for initial development
- **Split Distribution**:
  - S_rm_train: 400 tasks (40%)
  - S_rm_dev: 100 tasks (10%)
  - S_policy_train: 300 tasks (30%)
  - S_policy_dev: 100 tasks (10%)
  - S_final_test: 100 tasks (10%)
- **Verification**: No overlap detected between splits
- **File**: `configs/env/webshop_task_manifest_v0.1.0.json`

#### 2026-02-10: WebShop Data + Runtime Verification
- **Decision**: Keep WebShop as submodule and run through local wrapper with small split defaults
- **Validation**:
  - Verified data files in `third_party/WebShop/data/` including full set and small-split files
  - Smoke test passed via `src/env/webshop.py` (`reset` + `search` step)
- **Note**: Current local run path uses fallback search when `pyserini` is unavailable; this is acceptable for development/debug, while official benchmark runs should use full dependency setup

#### 2026-02-11: Baseline Reliability Hardening
- **Decision**: Upgrade single-agent baseline from placeholder to reliability-focused execution pipeline
- **Implementation**:
  - Added unified `BudgetState` envelope for `tok/call/step` accounting
  - Added two-stage action generation with constrained repair prompt
  - Added per-step structured `intermediate_outputs` logs (`llm_text`, `parsed`, attempts, budget before/after)
  - Added task-level deterministic seed derivation (`derive_task_seed`)
  - Added RM-BoN scoring fields (`rm_score`, `bon_score`, `bon_candidates`, `picked_index`)
  - Added minimal open-loop RM-MCTS skeleton with budget-aware planning
  - Added `fake` LLM provider for key-free deterministic smoke testing
- **Outcome**: Baseline stack moved from "can run locally" to "debuggable + reproducible + CI-verifiable"

#### 2026-02-11: CI Guardrail for Baselines
- **Decision**: Add required smoke checks for both normal and repair paths
- **Implementation**:
  - Added `scripts/smoke_baselines_fake.sh` (single-command smoke, artifact checks, schema field checks)
  - Added GitHub Actions workflow `.github/workflows/smoke-fake-baselines.yml`
  - Matrix checks:
    - `smoke (0.0)` for normal path
    - `smoke (0.25)` for repair path coverage
  - Artifact output isolated by matrix `OUT_ROOT`
- **Outcome**: baseline regressions are now machine-gated before merge

#### 2026-02-11: L0->L1 Real-Baseline Observability Anchors
- **Decision**: Add request/billing/behavior anchors before first real-provider dev-slice run
- **Implementation**:
  - Extended `src/eval/baselines.py` LLM attempt logs with request anchor fields (`provider/model/temperature/max_output_tokens/top_p/stop/reasoning_effort/tools_enabled`)
  - Added per-attempt billing fields (`prompt_tokens/completion_tokens/total_tokens`) and latency (`latency_ms`)
  - Added behavior fields (`fallback_used`, `fallback_reason`, repair/fallback counters, budget-cut flags)
  - Added `--max_tasks` support for small-slice runs and extended `metrics.json` with `avg_env_step` and observability rates
  - Added `scripts/run_real_dev_slice.sh` to run reproducible real-provider dev slices and emit `summary.json`
- **Outcome**: infra now supports low-cost L0->L1 runs with auditable request/cost/behavior traces

---

## Implementation Notes

### Environment (WebShop)
- Status: ✅ Data available, wrapper runnable
- Dependencies:
  - ✅ Core environment path working via `src/env/webshop.py`
  - ⚠️ Optional official components (`pyserini` + Lucene index build) are still recommended for benchmark-faithful retrieval
- Key config: `configs/env/webshop_v0.1.0.json`
- Canonical `setup_command` (must match `configs/env/webshop_v0.1.0.json`):
  ```bash
  ls third_party/WebShop/data/items_shuffle_1000.json third_party/WebShop/data/items_ins_v2_1000.json third_party/WebShop/data/items_human_ins.json && source venv/bin/activate && python -c 'from src.env.webshop import create_webshop_env; from src.env.base import Action; env=create_webshop_env({"max_steps":80,"num_products":1000,"observation_mode":"text"}); obs,_=env.reset(task_id="0"); step=env.step(Action(action_type="search", args={"query":"water bottle"})); print(len(obs.text),step.reward,step.done)'
  ```
- Implementation: `src/env/webshop.py` verified with smoke test

### Reward Model
- Status: Not started
- Approach: Offline training from trajectories
- Key challenge: Hard negative generation

### Orchestrator
- Status: Not started
- Algorithm: GRPO (Group Relative Policy Optimization)
- Key challenge: Parse stability + KL control
  
---

## Experimental Results

### Baseline Results
#### 2026-02-11 Offline Smoke (fake provider, 1-task split)
- Methods: `greedy`, `rm_bon`, `rm_mcts`
- Command: `./scripts/smoke_baselines_fake.sh`
- Status: ✅ Passed
- Artifacts:
  - `results/smoke_fake/<run_id>/0.0/`
  - `results/smoke_fake/<run_id>/0.25/`
- Purpose: regression/sanity only (not reportable benchmark numbers)

### RM Training
*To be filled*

### Orchestrator Training
*To be filled*

---

## Issues and Resolutions

### Issue #1: WebShop Installation Method
- **Date**: 2026-02-10
- **Description**: WebShop is not available as a standard pip package
- **Resolution**: Used git submodule approach, installed core dependencies (gym, spacy, faiss-cpu)
- **Status**: ✅ Resolved
- **Next Step**: Run first baseline rollout using the verified wrapper path

### Issue #2: Baseline Runtime/Interface Mismatch
- **Date**: 2026-02-11
- **Description**:
  - `task_id` manifest format (`task_00001`) mismatched env reset parser
  - baseline CLI/package imports had blocking mismatches
  - run script argument mismatch and false-success pipe behavior
- **Resolution**:
  - normalized task-id parsing in `src/env/webshop.py`
  - fixed eval/utils import surfaces and CLI compatibility
  - fixed `run_baselines.sh` arg forwarding + `pipefail` handling
- **Status**: ✅ Resolved

---

## Next Steps (Immediate)

1. **Run First Real-LLM Baseline (small dev slice)**:
   - Use `S_policy_dev` (or 20-task subset) with real API keys
   - Export `OPENAI_API_KEY` and run greedy/sampling/rm_bon/rm_mcts
   - Save comparison table: success, avg_tok, avg_call, avg_step

2. **Lock Branch Protection Required Checks**:
   - Require `smoke (0.0)` + `smoke (0.25)` before merge

3. **Start RM Data Collection** (Week 3-4):
   - Collect trajectories from simple policies
   - Implement hard negative generation
   - Train initial RM

---

## Progress Summary

**Week 1 Progress**: ~92% complete
- ✅ Project structure and configs
- ✅ Python environment setup
- ✅ Core code implementation (env, baselines, utils)
- ✅ Task splits generated
- ✅ WebShop data download and wrapper smoke test
- ✅ LLM integration for baseline policies
- ✅ Baseline reliability hardening (budget/repair/logging/seed)
- ✅ Offline fake smoke run + CI guardrails
- ⏳ First real-LLM baseline run

**Estimated Time to First Real Baseline Result**: <1 day
**Current Blocker**: real API execution + benchmark-faithful dependency path (`pyserini`/index) for official runs
