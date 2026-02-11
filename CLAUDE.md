# NextStep - Claude AI Collaboration Guide

## Project Overview

**NextStep** v0.1.0 is a research project training multi-agent orchestrators using reward models for unverifiable tasks. Target: AAAI 2027.

**Status**: Week 1/12 | **Phase**: Environment Setup + Baselines

---

## Quick Context

### Research Goal
Train an orchestrator `O_θ` that:
- Generates multi-agent workflows in JSON format
- Uses learned RM as reward (no ground truth success during training)
- Optimizes success-cost Pareto with anti-hacking mechanisms

### Key Constraints (Frozen v0.1.0)
- **Budget**: B_tok=6000, B_step=80, B_call=12, T_max=6
- **Splits**: S_rm_train, S_rm_dev, S_policy_train, S_policy_dev, S_final_test
- **No leakage**: Test set never used for training/selection

---

## Working Directory

```
/Users/xiangruichao/Desktop/NextStep
```

### Key Files
- `docs/research_spec.md`: Complete research design (frozen)
- `docs/notes.md`: Ongoing research notes
- `docs/failure_cases.md`: Failure case documentation
- `configs/`: All v0.1.0 configurations (frozen)

---

## Current Status

### ✅ Completed
- [x] Project structure created
- [x] All v0.1.0 config templates added
- [x] Python package structure initialized
- [x] WebShop integration + environment wrapper implementation
- [x] WebShop data verification and wrapper smoke test
- [x] Task split generation
- [x] Single-agent baselines implementation (greedy/sampling + RM-BoN/RM-MCTS skeleton)
- [x] Baseline reliability hardening (budget envelope, repair path, structured trajectory logs)
- [x] Offline fake baseline smoke + CI workflow (`smoke (0.0)`, `smoke (0.25)`)

### 🚧 In Progress (Week 1-2)
- [ ] First baseline experiment run

### 📋 Next Steps
1. Run first real-LLM baseline on a small dev slice (e.g., 20 tasks)
2. Compare greedy/sampling/RM-BoN/RM-MCTS with unified cost metrics
3. Start RM data collection and hard-negative pipeline
4. Execute first end-to-end evaluation pass

---

## Coding Guidelines

### File Organization
```
src/
├── env/          # Environment wrappers (BaseEnv, WebShopEnv)
├── rm/           # Reward model (BaseRM, RMTrainer, Ensemble)
├── orchestrator/ # Orchestrator (BaseOrch, GRPOTrainer, WorkerPool)
├── eval/         # Evaluation (metrics, baselines)
└── utils/        # Config loading, logging, trajectory utils
```

### Code Style
- **Language**: Python 3.10+
- **Style**: Follow Black (100 char line limit)
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style for all modules/functions

### Config Management
- **Never** hardcode hyperparameters in code
- Always load from `configs/` directory
- Use `src.utils.config.load_config()`
- Log loaded config to `run.json` for every experiment

### Logging Requirements
Every experiment must log:
- `run.json`: Full experiment metadata
- `trajectories.jsonl`: Episode trajectories
- `metrics.json`: Computed metrics
- `log.txt`: Full execution log

---

## Common Tasks

### Adding New Environment
1. Create `src/env/<name>.py`
2. Implement `BaseEnvironment` interface
3. Add config `configs/env/<name>_v0.1.0.json`
4. Update `src/env/__init__.py`

### Adding New Baseline
1. Implement in `src/eval/baselines.py`
2. Add to `scripts/run_baselines.sh`
3. Document in `docs/notes.md`

### Adding New Metric
1. Implement in `src/eval/metrics.py`
2. Update evaluation scripts
3. Add to main results table template

---

## Experiment Protocol

### Before Running
1. Ensure all configs are committed to git
2. Verify task manifests are frozen
3. Check no test leakage in code
4. Set `EXP_ID` with timestamp

### During Run
1. Monitor `parse_fail` rate (should decrease)
2. Watch KL divergence (should stay stable)
3. Check cost distribution (no explosions)

### After Run
1. Verify `run.json` is complete
2. Check trajectories are saved
3. Compute metrics and update `docs/notes.md`
4. Document any failures in `docs/failure_cases.md`

---

## Troubleshooting

### Issue: High Parse Failure Rate
**Solution**:
- Check workflow format constraints
- Add more few-shot examples
- Tighten JSON schema validation
- Consider format-SFT pre-training

### Issue: RM Hacking
**Solution**:
- Increase hard negative ratio
- Enable online mining
- Add more rule anchors
- Use more conservative ensemble (min)

### Issue: Cost Explosion
**Solution**:
- Increase `lambda_c` cost penalty
- Enable hard budget truncation
- Add early stopping heuristics
- Reduce max workflow steps

---

## Git Workflow

### Branching
- `main`: Stable commits only
- `feature/*`: Feature development
- `fix/*`: Bug fixes
- `exp/*`: Experiment branches

### Commit Messages
Format: `[PREFIX] Description`

Prefixes:
- `[ENV]`: Environment changes
- `[RM]`: Reward model changes
- `[ORCH]`: Orchestrator changes
- `[EVAL]`: Evaluation changes
- `[DOC]`: Documentation
- `[EXP]`: Experiment results

### Before Pushing
1. Run tests (if implemented)
2. Check no test leakage
3. Verify configs are frozen
4. Update `docs/notes.md`

---

## Research Milestones (12 Weeks)

| Week | Goal | Status |
|------|------|--------|
| 1-2 | Environment + baselines | 🚧 In progress |
| 3-4 | RM pipeline | ⏳ Not started |
| 5-6 | Orchestrator format stability | ⏳ Not started |
| 7-8 | Orchestrator RL + Pareto | ⏳ Not started |
| 9 | RM ensemble + hacking probes | ⏳ Not started |
| 10 | Online mining loop | ⏳ Not started |
| 11 | Worker pool shift + ablations | ⏳ Not started |
| 12 | Finalize figures/tables + writing | ⏳ Not started |

---

## References

See `关键论文/` for:
- ARMAP: Reward modeling and planning
- GPTSwarm: Graph-based agent orchestration
- The Conductor: Learning to orchestrate

---

## Quick Commands

```bash
# Activate environment
source venv/bin/activate

# Run baseline
./scripts/run_baselines.sh --method greedy --seed 42

# Train RM
./scripts/train_rm.sh --config configs/rm/rm_training_v0.1.0.json

# Train orchestrator
./scripts/train_orchestrator.sh --rm-checkpoint artifacts/models/rm_ensemble

# Evaluate
./scripts/eval_all.sh --checkpoint artifacts/models/orchestrator_final

# Check logs
tail -f results/exp_logs/*/log.txt

# View project tree
tree -L 3 -I 'venv|__pycache__|*.pyc'
```

---

*Last Updated: 2026-02-11*
