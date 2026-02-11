# Research Spec (v0.1.0)

Status: Frozen for v0.1.0.  
Date: 2026-02-10  
Owner: NextStep

## 0. One-liner

Train an orchestrator policy that generates multi-agent workflows using learned reward models (RM) as training signal on unverifiable / weakly-verifiable interactive tasks, optimizing success-cost Pareto while resisting reward hacking.

---

## 1. Motivation & Gap

### 1.1 Problem

In interactive environments, ground-truth correctness is often unavailable or expensive (open-ended objectives, partial observability, long-horizon actions). Existing orchestration learning methods commonly assume verifiable rewards, limiting applicability.

### 1.2 Proposed idea

Replace hard correctness reward with an automatically trained reward model (trajectory-level utility estimator) and use it to train an orchestrator via RL (plus cost regularization), optionally with test-time workflow search.

### 1.3 Key risks (must be addressed)

- Reward hacking / gaming the RM
- Credit assignment over long-horizon workflows
- Distribution shift (RM trained on synthetic/off-policy data vs. on-policy orchestration trajectories)
- Cost blow-up (orchestrator learns to spam expensive workers)

---

## 2. Research Questions (RQs) & Hypotheses (Hs)

### RQ1: Efficacy

Can RM-guided orchestration improve task success on unverifiable interactive tasks under fixed budget vs. strong baselines?

H1: Under the same budget envelope (tokens/steps/calls), RM-guided orchestration achieves higher success than:

- single-agent RM-planning (best-of-N, MCTS)
- fixed-topology multi-agent pipelines (static roles)

### RQ2: Efficiency

Can it improve success-cost Pareto?

H2: For any target success threshold, RM-guided orchestration achieves lower average cost.

### RQ3: Robustness to worker pool shift

Can it generalize when the worker pool changes?

H3: Performance drop under worker pool shift is smaller than fixed-topology baselines.

### RQ4: Anti-hacking

Can we reduce RM gaming while maintaining performance?

H4: RM ensemble + online hard-negative mining reduces RM-high-but-env-fail rate without losing Pareto advantage.

---

## 3. Scope & Non-goals

### In scope

- Interactive benchmarks with official success evaluation
- Learned RM + orchestrator RL + cost-aware optimization
- Robustness tests and adversarial hacking probes
- Reproducible pipelines, frozen configs, logged artifacts

### Out of scope (for v0.1)

- Human-in-the-loop reward labeling beyond small calibration sets
- Real web browsing outside benchmarks
- Training worker LLMs (workers fixed; only orchestrator and RM trained)
- Full theoretical convergence proofs

---

## 4. Definitions & Notation

- Task/query: q
- Environment: E
- Workflow (orchestration plan): w = [s_1...s_T], each step has:
  - `subtask_t` (natural language)
  - `worker_id_t`
  - `access_list_t` (which prior outputs are visible)
- Trajectory: tau = environment interaction trace + intermediate agent outputs induced by executing w
- Reward model: RM(q, tau) in [0, 1]
- Cost:
  - raw token cost: `tok(tau)`
  - raw env-step cost: `step(tau)`
  - raw API-call cost: `call(tau)`
  - normalized cost:
    - `tok_n = tok(tau) / B_tok`
    - `step_n = step(tau) / B_step`
    - `call_n = call(tau) / B_call`
  - total: `C(tau) = alpha * tok_n + beta * step_n + gamma * call_n`
- Total reward:
  - `R = parse_ok * RM_ens(q, tau) - lambda_c * C(tau) - lambda_l * T`

Default v0.1.0 cost weights: `alpha=0.6, beta=0.3, gamma=0.1`.

---

## 5. Datasets / Environments

### 5.1 Benchmarks (v0.1.0)

Primary:

- WebShop (Phase-1 only, frozen for v0.1.0 main table)

Secondary (v0.1.x extension):

- ALFWorld or ScienceWorld

Environment/version freeze (required):

- `ENV_VERSION`: store benchmark source commit, data checksum, and wrapper version in:
  - `configs/env/webshop_v0.1.0.json`
- `TASK_ID_MANIFEST`: frozen task id lists for `S_rm_train`, `S_policy_train`, `S_policy_dev`, `S_final_test` in:
  - include `S_rm_dev` and `S_policy_dev` explicitly as task-id sets
  - `configs/env/webshop_task_manifest_v0.1.0.json`
- Action-space policy freeze:
  - if action pruning is enabled, save exact pruning rules and parameters in `configs/env/webshop_v0.1.0.json`

### 5.2 Splits and leakage control (frozen)

Define five task-id sets:

- `S_rm_train`: for RM data generation/training only
- `S_rm_dev`: RM model selection/calibration only
- `S_policy_train`: for orchestrator RL rollouts and online mining only
- `S_policy_dev`: policy dev tuning and `tau_hack` quantile estimation only
- `S_final_test`: final reporting only

Rules:

- `S_final_test` is never used for RM training, online mining, or hyperparameter selection.
- `S_rm_dev` must not overlap with `S_rm_train`, `S_policy_train`, `S_policy_dev`, or `S_final_test`.
- `S_policy_dev` must not overlap with `S_policy_train` or `S_final_test`.
- Any sample in online mining must come from `S_policy_train`.
- `S_policy_dev` and `S_rm_dev` must be persisted in `TASK_ID_MANIFEST` (not generated on the fly).
- Probe sets are frozen in `configs/env/webshop_hacking_probe_manifest_v0.1.0.json` with named sets:
  - `S_probe_train` (from `S_policy_train`)
  - `S_probe_dev` (from `S_policy_dev`)
- Probe sets are diagnostics-only and never used for final model selection.

### 5.3 Unverifiable training rule

During orchestrator RL, do not use environment success as reward.  
Environment success may be logged for diagnostics, but must be marked `debug_only=true` and excluded from optimizer inputs.
Environment success is allowed for RM data construction/refresh on `S_rm_train` and `S_policy_train`, but never allowed in policy reward.

### 5.4 RM label policy (allowed usage)

Benchmark success can be used to label RM training examples inside `S_rm_train` only.  
Strong-judge labels are allowed only inside `S_rm_train`.

Strong judge spec (frozen, required when used):

- `JUDGE_MODEL_ID`: exact model name/version/date
- `JUDGE_TEMPERATURE`: default 0.0
- `JUDGE_PROMPT_TEMPLATE`: saved prompt template path under `configs/rm/`
- `JUDGE_VOTING_RULE`: single-judge or multi-judge majority (default: 3 judges, majority vote)
- `JUDGE_COST_CAP`: max judge calls/tokens per sample
- All judge metadata must be logged to `run.json`.

---

## 6. System Components

### 6.1 Orchestrator (trainable)

- Model: `O_theta`
- Output format: strict JSON (required schema)
- Parse gating:
  - `parse_ok=0`: set `R=0`, log `parse_fail`, skip execution
  - `parse_ok=1`: execute workflow, compute RM/cost

### 6.2 Worker pool (fixed)

- Pool size (v0.1.0): 4 workers
- Training-time randomization:
  - sample 3/4 workers each episode to improve shift robustness
- Worker pool identities must be frozen in:
  - `configs/orchestrator/worker_pool_v0.1.0.json`

### 6.3 Reward model (trainable)

- Model: `RM_phi`
- Inputs: query q + trajectory summary
- Output: scalar utility in [0,1]
- v0.1.0 uses single-head RM; cost remains outside RM in objective.

TrajectorySummarySchema (frozen v0.1.0):

- `query_text`
- `workflow_steps`: list of `{subtask, worker_id, access_list}`
- `step_outputs`: truncated worker outputs per step
  - max chars per step output: 600
  - max steps retained: `T_max`
- `env_trace_summary`: ordered `(obs_t, action_t)` tuples (text-truncated)
  - max env trace entries: 80
  - max chars per obs/action field: 240
- `terminal_output`
- `budget_flags`: parse_fail/timeout/budget_truncated

Serialization rule:

- deterministic JSON key order
- truncation happens before RM tokenization
- redact secrets/keys if present

### 6.4 Planner (optional)

- Workflow Best-of-N: sample N workflows from `O_theta`, execute, pick best RM-adjusted score
- Workflow MCTS: node=partial workflow, edge=append step, value=RM estimate or rollout return

---

## 7. Reward Model Data Pipeline

### 7.1 Data sources

Off-policy trajectories from:

- scripted/simple policies
- single-agent baseline runs
- early orchestrator snapshots

All from `S_rm_train` only.

### 7.2 Positive examples

Trajectories satisfying intent via benchmark success or strong-judge labels (within `S_rm_train` only).

### 7.3 Hard negative construction (critical)

For each positive `tau+`, create 1-3 `tau-`:

- remove key action / break critical subgoal
- wrong item / wrong terminal output
- plausible but wrong trajectories likely to fool RM

NegGen-v0.1 protocol (frozen):

- Mix ratio:
  - 50% rule-based edits (deterministic templates)
  - 50% model-based rewrites (LLM rewrite with fixed prompt)
- Rule-based edits include:
  - drop critical step
  - replace key entity/item
  - terminate one step early
- Model-based rewrites include:
  - fluent but goal-violating paraphrase
  - subtle contradiction with required outcome
- Noise filter:
  - run negative replay/judge check; discard candidate if marked successful
  - keep only negatives with confidence flag `neg_valid=true`

### 7.4 RM training objective

- Primary loss: pairwise ranking on `(tau+, tau-)`
- RM-dev metrics:
  - pairwise AUC
  - pairwise accuracy
  - RM-success Spearman correlation on held-out `S_rm_dev`

### 7.5 RM ensemble (anti-hacking)

- Train K=3 RMs with different seeds/shuffles
- Aggregate with conservative rule:
  - `RM_ens = min_k RM_k`
- Diagnostics-only aggregate:
  - log `RM_mean = mean_k RM_k` for analysis plots (not used for optimization)

### 7.6 Online hard-negative mining (v0.2 feature, pre-specified now)

Every N policy iterations:

- sample on-policy trajectories from `S_policy_train`
- collect samples satisfying hacking trigger:
  - `RM_ens >= q80(RM_ens on S_policy_dev)` and `env_success=0`
- add to RM hard-negative buffer
- optional RM refresh with capped update steps

Refresh caps (frozen defaults):

- `REFRESH_FREQUENCY_N=10` policy iterations
- `BUFFER_SIZE_CAP=50000` trajectories
- `RM_REFRESH_MAX_STEPS_PER_CYCLE=400`
- `RM_REFRESH_LR_MULT=0.5` (relative to offline RM LR)

---

## 8. Orchestrator Training

### 8.1 Initialization

- Prompt + few-shot format examples
- Optional format-only SFT

### 8.2 RL algorithm

- GRPO-style grouped online RL + KL to reference
- For each query q:
  - sample K workflows from `O_theta(.|q)`
  - execute and score each with `R_i`
  - normalize advantages within group
  - update theta

RL_HPARAMS_MIN (frozen defaults):

- `reference_policy`: format-SFT checkpoint at iteration 0 (`orchestrator_ref_v0.1.0`)
- `optimizer`: AdamW
- `lr`: 1e-5
- `batch_queries_per_update`: 4
- `adv_norm`: z-score within group
- `adv_clip`: [-5, 5]
- `grad_clip_norm`: 1.0

### 8.3 Objective

`R_i = parse_ok * RM_ens(q, tau_i) - lambda_c * C(tau_i) - lambda_l * len(w_i)`

### 8.4 Budget envelope (frozen v0.1.0)

- `T_max` (max workflow steps): 6
- max env steps per episode (`B_step`): 80
- max tokens per worker call: 512
- max worker calls per episode (`B_call`): 12
- rollout count K: 16 (main), 32 (scaling ablation)
- max total tokens per episode (`B_tok`): 6000

Any method exceeding envelope is truncated and flagged `budget_truncated=true`.

TruncationPolicy (frozen):

- token budget exceeded:
  - stop current generation immediately, keep partial text, mark step `token_truncated=true`
  - no new worker calls after exceed
- env step budget exceeded:
  - terminate episode with `timeout_env_budget=true`
- worker call budget exceeded:
  - reject further calls, end workflow execution, mark `call_budget_exceeded=true`

Parallelism policy (frozen):

- `MAX_PARALLEL_CALLS=2` per workflow step
- rollouts can execute concurrently for system throughput, but per-episode cost accounting is independent of runtime concurrency
- parallel calls count fully toward `call(tau)` and generated-token totals

### 8.5 KL / stability

- Keep KL penalty active in all policy updates
- Log per-update KL, parse-fail rate, reward variance

KL defaults:

- `kl_coef_init=0.02`
- `kl_coef_schedule`: linear warmup to 0.05 by 30% training, then constant

---

## 9. Baselines (must implement)

### 9.1 Single-agent

- Greedy / Sampling
- RM-Best-of-N
- RM-MCTS

### 9.2 Fixed-topology multi-agent

- parallel attempts + vote
- static planner->executor
- fixed debate/critique (2 rounds)

### 9.3 Learning-based controls

- Orchestrator RL with:
  - no RM (cost-only)
  - weak heuristic reward (rule-only)
  - RM single vs RM ensemble

### 9.4 Fairness protocol (frozen)

All methods must use:

- same worker pool candidates for the experiment
- same budget envelope (`B_tok`, `B_step`, `B_call`, `T_max`)
- same maximum parallelism (`MAX_PARALLEL_CALLS=2`)
- same RM access rule:
  - if baseline uses RM (RM-BoN/RM-MCTS), it must use the same RM ensemble checkpoint as main method in that experiment group

---

## 10. Ablations (minimum set)

1. w/o RM (heuristic only)
2. RM single vs RM ensemble
3. w/o cost penalty
4. w/o access list (full history vs restricted access)
5. offline RM vs online RM updates
6. worker pool shift (train pool A; test pool B with replacement)

Worker shift protocol (frozen):

- Pool A size 4, Pool B size 4
- replace ratio: 1/4 and 2/4
- report absolute drop and relative drop vs in-pool test

---

## 11. Metrics & Reporting

### 11.1 Primary metrics

- Success rate (official env metric)
- Average cost (`tok`, `step`, `call`, and composite `C`)
- Pareto curve: success vs budget

Pareto budget definition (frozen):

- Pareto points vary `B_tok` only.
- `B_step` and `B_call` are fixed at v0.1.0 constants for all Pareto points.
- For v0.1.0: `B_step=80`, `B_call=12` for budgets `{2000, 4000, 6000}`.

### 11.2 Robustness / safety metrics

- Reward hacking rate:
  - `P(RM_ens >= tau_hack and env_fail)`
  - `tau_hack` fixed as 80th percentile of RM_ens on `S_policy_dev` computed once per experiment
- RM calibration:
  - reliability bins (10 bins) of RM_ens vs empirical success
- Seed variance

### 11.3 Statistical plan (frozen)

- Seeds: >=3 per setting
- CI: bootstrap 95% on test tasks
- Paired comparisons on identical task ids
- Multiple testing correction:
  - primary endpoints: Holm-Bonferroni
  - secondary endpoints: Benjamini-Hochberg (FDR 0.05)

---

## 12. Experiment Matrix

### 12.1 Fixed constants (v0.1.0)

- env: WebShop
- worker pool size: 4
- `T_max=6`
- `B_step=80`
- `B_tok=6000`
- `B_call=12`
- max tokens/call=512
- K in `{16, 32}`

### 12.2 Grid

E1 Baseline reproduction:

- methods: greedy, sampling, RM-BoN, RM-MCTS

E2 RM training:

- data size: {small, medium}
- negative type: {easy, hard}
- report AUC + calibration + Spearman

E3 Orchestrator RL:

- RM: single
- `lambda_c`: {0.05, 0.1}
- `lambda_l`: {0.01}
- output Pareto

E4 Anti-hacking:

- RM ensemble: on/off
- online mining: on/off
- report hacking rate and success-cost

E5 Worker pool shift:

- train pool A, test pool B (1/4 and 2/4 replacement)
- report performance drop

Main table (frozen v0.1.0):

- Methods:
  - Greedy
  - RM-BoN
  - RM-MCTS
  - Static planner->executor
  - Ours (RM-single)
  - Ours (RM-ensemble)

Primary endpoint (frozen):

- `Success@B_tok=6000` on `S_final_test`
- Secondary headline:
  - Pareto-AUC over budgets `{2000, 4000, 6000}`
- Pareto-AUC budget policy:
  - `B_tok` in `{2000, 4000, 6000}`
  - `B_step` fixed to `80`
  - `B_call` fixed to `12`

---

## 13. Logging & Artifacts

### 13.1 Directory structure (frozen)

```text
repo/
  research_spec.md
  CLAUDE.md
  configs/
    env/
    rm/
    orchestrator/
  data/
    rm/
      raw/
      processed/
  src/
    env/
    rm/
    orchestrator/
    eval/
  scripts/
    run_baselines.sh
    train_rm.sh
    train_orchestrator.sh
    eval_all.sh
  results/
    exp_logs/
    metrics/
    plots/
  artifacts/
    models/
    snapshots/
  docs/
    notes.md
    failure_cases.md
    hacking_probe_spec.md
    patches/
      patch_YYYYMMDD.md
```

### 13.2 Run manifest (required each run)

`results/exp_logs/<exp_id>/run.json` must include:

- git commit hash
- timestamp
- environment + split ids
- model versions (`O_theta`, `RM_phi`, worker pool ids)
- worker pool config path + checksum:
  - `configs/orchestrator/worker_pool_v0.1.0.json`
- hyperparameters (`lambda_c`, `lambda_l`, K, KL coef, `T_max`, budgets)
- random seeds
- hardware/runtime

### 13.3 Trajectory log schema (JSONL)

Per episode fields:

- `qid`, `query_text`
- `split_id` (`S_rm_train` / `S_policy_train` / `S_final_test`)
- `workflow` (steps with subtask/worker_id/access_list)
- `intermediate_outputs` (truncated)
- `env_trace`
- `rm_scores` (each RM + aggregate)
  - required fields: `rm_ens_min`, `rm_mean`
- `cost` (`tok`, `step`, `call`, `C`)
- `env_success`
- `debug_only` (bool)
- `flags` (`parse_fail`, `timeout`, `budget_truncated`, `policy_violation`)

---

## 14. Milestones (12-week)

- W1-2: environment + single-agent baselines
- W3-4: RM pipeline + RM metrics
- W5-6: orchestrator format stability + optional format SFT
- W7-8: orchestrator RL (offline RM) + first Pareto
- W9: RM ensemble + hacking probes
- W10: online mining loop
- W11: worker pool shift + ablations
- W12: finalize figures/tables + writing

---

## 15. Risk Register & Mitigations

R1 Reward hacking:

- conservative RM ensemble (`min`)
- hard negatives + online mining
- rule anchors

Rule anchors (frozen minimum set):

- `ANCHOR_1`: terminal output must include final answer field with required entity (for WebShop: selected item id/title)
- `ANCHOR_2`: trajectory must contain at least one goal-directed action from allowed critical-action list
- `ANCHOR_3`: terminal output must include explicit completion/status statement
- Anchor config file:
  - `configs/rm/anchors_v0.1.0.json`

R2 Training instability:

- KL regularization
- grouped advantage normalization
- caps on workflow/env steps

R3 RM-policy distribution shift:

- periodic on-policy refresh from `S_policy_train`
- strict held-out RM-dev

R4 Cost explosion:

- explicit cost penalty
- hard budget truncation
- early stop/pruning

---

## 16. Reproducibility Checklist

- All configs committed and immutable per experiment id
- No test leakage into any training loop
- Seeds recorded; >=3 seeds on primary results
- Save and version RM/orchestrator checkpoints for final release (if policy allows)

---

## 17. Patch Process

This file is frozen at v0.1.0.  
Changes must be added to:

- `docs/patches/patch_YYYYMMDD.md`

Each patch must include:

- version bump target
- what changed
- why it changed
- which experiments are affected
