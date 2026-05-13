# Pipeline improvement roadmap

This roadmap lists practical, high-impact improvements for the fairness pipeline in `src/`.

## 1) Add experiment tracking + lineage (highest priority)

**Why:** The workflow already exports many CSV/figures and a config snapshot, but it does not yet provide a single queryable experiment registry across runs.

**Improvements:**
- Add run IDs and parent-child links (baseline -> mitigation runs).
- Log all metrics/tables to MLflow (or Weights & Biases) with:
  - git commit hash
  - dataset hash/signature
  - config JSON
  - package lockfile snapshot
- Store champion-selection rationale as structured artifacts (not only report text).

**Outcome:** reproducibility, easy cross-run comparison, auditability.

## 2) Formalize leakage and temporal validation gates

**Why:** You already mention leakage checks and time-aware splits in documentation. Making these enforceable gates will prevent accidental misuse.

**Improvements:**
- Implement a preflight validator that fails fast when:
  - target is constant/missing
  - train/test split contains duplicate IDs across partitions
  - post-outcome timestamp features are included
- Add a `split_strategy` config (`random_stratified`, `group`, `temporal`).
- Require temporal split for datasets with event-time columns.

**Outcome:** stronger methodological validity and less optimistic bias.

## 3) Add probability calibration as a first-class stage

**Why:** Fairness thresholding quality depends heavily on calibrated probabilities.

**Improvements:**
- Add model calibration options (`isotonic`, `platt`) fitted on validation folds.
- Track pre/post calibration Brier score and ECE.
- Optionally run mitigation on calibrated probabilities only.

**Outcome:** more stable threshold optimization and better clinical interpretability.

## 4) Expand fairness uncertainty quantification

**Why:** You already use bootstrap significance; extend this to decision reporting.

**Improvements:**
- Report confidence intervals for all group-level rates, not only gap summaries.
- Add multiple-comparison adjustment for many subgroup/intersection tests.
- Add minimum detectable effect (MDE) guidance based on subgroup sample size.

**Outcome:** fewer false alarms and clearer uncertainty communication.

## 5) Introduce explicit governance policy checks in code

**Why:** Governance controls are documented, but they should be machine-enforced.

**Improvements:**
- Add configurable pass/fail guardrails:
  - max allowed balanced-accuracy drop
  - max allowed fairness gap(s)
  - minimum subgroup support
- Add a `deployment_readiness` table: `pass/fail + reasons`.
- Fail CI when a configured guardrail fails.

**Outcome:** policy-to-code alignment and safer model promotion.

## 6) Strengthen intersectional scalability controls

**Why:** The config supports broad intersection generation; combinatorics can explode.

**Improvements:**
- Add adaptive pruning before mitigation jobs:
  - skip intersections below support threshold
  - prioritize intersections with worst baseline gaps
- Cache intermediate encoded matrices reused across mitigation jobs.
- Add runtime budget mode (e.g., stop after N minutes with best-so-far report).

**Outcome:** faster runs with better use of compute budget.

## 7) Add drift and fairness monitoring package for post-training use

**Why:** Current workflow is run-centric; monitoring in production/post-hoc should be standardized.

**Improvements:**
- Export a monitoring spec (YAML/JSON) with:
  - protected attributes
  - thresholds for drift/fairness alerts
  - retraining triggers
- Generate monthly-ready metrics schema for dashboards.

**Outcome:** operational continuity from offline validation to ongoing oversight.

## 8) CI/CD hardening for reproducibility

**Why:** A pipeline this broad benefits from automated checks at pull request time.

**Improvements:**
- Add CI matrix for Python versions + optional fairness backends availability.
- Add smoke-test dataset fixtures and deterministic regression tests.
- Add a lockfile strategy (`pip-tools`/`uv`) and verify dependency integrity.

**Outcome:** fewer breakages, consistent runtime behavior.

## Suggested implementation order (90-day plan)

1. **Weeks 1–2:** governance policy checks + preflight leakage/validation gates.
2. **Weeks 3–5:** experiment tracking + lineage.
3. **Weeks 6–7:** calibration stage + additional uncertainty outputs.
4. **Weeks 8–10:** CI hardening and reproducibility checks.
5. **Weeks 11–12:** intersectional scaling optimizations + monitoring export spec.

---

## Quick wins you can implement immediately

- Add `split_strategy` to config and validate required columns for temporal splits.
- Add a `deployment_readiness.csv` summary produced at the end of every run.
- Persist a single `run_metadata.json` containing config, hashes, seeds, and git SHA.
- Add one CI smoke test that runs `execution_preset="smoke"` on a tiny synthetic dataset.
