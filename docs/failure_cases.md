# Failure Cases and Analysis (NextStep v0.1.0)

## Purpose

Document and analyze failure cases to understand limitations and guide improvements.

---

## Categories

### 1. Reward Hacking Cases

#### Case #1: [ID]
- **Date**: YYYY-MM-DD
- **Method**: [Which method/baseline]
- **RM Score**: [High/Low]
- **Env Success**: [0/1]
- **Description**: [What happened]
- **Root Cause**: [Why RM was fooled]
- **Fix**: [What was done to address]

### 2. Parse Failures

#### Case #1: [ID]
- **Date**: YYYY-MM-DD
- **Iteration**: [Training iteration]
- **Query**: [Task query]
- **Generated Output**: [Invalid JSON or wrong format]
- **Root Cause**: [Format violation, truncation, etc.]
- **Fix**: [Prompt engineering, format constraints, etc.]

### 3. Cost Explosion

#### Case #1: [ID]
- **Date**: YYYY-MM-DD
- **Budget Limit**: [Expected]
- **Actual Cost**: [Observed]
- **Breakdown**: [Token/Step/Call costs]
- **Root Cause**: [Why cost exceeded]
- **Fix**: [Cost penalty, truncation, etc.]

### 4. Distribution Shift

#### Case #1: [ID]
- **Date**: YYYY-MM-DD
- **RM Train Domain**: [Where RM was trained]
- **Test Domain**: [Where it failed]
- **Performance Drop**: [Quantify]
- **Root Cause**: [Distribution mismatch]
- **Fix**: [Domain adaptation, online mining, etc.]

---

## Patterns and Insights

### Common Failure Modes
*To be filled as experiments progress*

### Mitigation Strategies
*To be filled as experiments progress*

---

## Action Items

- [ ] Regularly update this document during experiments
- [ ] Categorize failures by type
- [ ] Identify systemic issues vs. one-off problems
- [ ] Track which mitigations are effective
