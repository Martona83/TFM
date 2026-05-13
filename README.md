# TFM Fairness in Biomedical AI

## 1. Project title
**TFM: Fairness, Bias Evaluation, and Machine Learning in Biomedical/Clinical AI**

## 2. Short description
This repository contains code and notebook workflows for a Master's Thesis (TFM) focused on fairness and bias evaluation in biomedical/clinical machine learning. It is structured for reproducibility and safe sharing without exposing private or sensitive data.

## 3. Repository structure
```
tfm-fairness-biomedical-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
├── src/
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── outputs/
├── reports/
└── docs/
```

## 4. Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. How to run the notebook
1. Place notebook files in `notebooks/`.
2. Put local datasets in `data/raw/`.
3. Launch Jupyter:
   ```bash
   jupyter lab
   ```
4. Open the target notebook in `notebooks/`.

If imports from `src/` fail in a notebook, add this setup cell:
```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().resolve().parents[0]))
```

## 6. Data availability and data placement
- Raw/clinical/private datasets are **not included**.
- Store local input data in `data/raw/`.
- Store cleaned or transformed local data in `data/processed/`.
- Add only explicitly safe, non-sensitive, redistributable sample data if needed.

## 7. Reproducibility notes
- Prefer fixed random seeds (`random_state` / `seed`) in training, splitting, and evaluation steps.
- Record package versions and experiment configuration before final thesis runs.
- Generated artifacts (models, checkpoints, outputs) are excluded from version control by default.

## 8. Main dependencies
See `requirements.txt`. Initial baseline includes common data-science and fairness libraries.

## 9. Ethical and privacy note
This repository is intended for academic reproducibility. Do not upload personal health information (PHI), credentials, or sensitive clinical data.

## 10. Citation / academic use note
If you reuse this code in academic work, cite the thesis and this repository version/commit hash.

## 11. Trustworthy fairness audit pipeline (for any CSV/dataset)
Use this checklist-oriented pipeline for every new dataset, including ad-hoc CSV files:

1. **Ingest and schema validation**
   - Load CSV with explicit `dtype`/date parsing when possible.
   - Verify target column, identifier columns, and candidate sensitive attributes.
   - Reject runs if target is missing/constant or if sensitive columns are empty.

2. **Data quality and leakage controls**
   - Run missingness, duplicate, and outlier checks.
   - Detect likely leakage features (post-outcome fields, IDs, timestamps after event).
   - Split train/validation/test before target-aware transformations.

3. **Baseline modeling with reproducible settings**
   - Use fixed seeds and stratified splits.
   - Train multiple baselines (e.g., logistic regression, random forest, gradient boosting).
   - Report discrimination + calibration together (ROC-AUC, PR-AUC, balanced accuracy, ECE/Brier).

4. **Fairness audit across all relevant slices**
   - Audit each sensitive attribute separately and in intersections (e.g., sex×age_group, race×sex).
   - Compute gap metrics (FPR/FNR/TPR/selection-rate/accuracy) with confidence intervals.
   - Include support counts per group and flag low-support groups.

5. **Mitigation experiments**
   - Evaluate pre-, in-, and post-processing options under the same split protocol.
   - Compare to baseline using both fairness improvement and performance guardrails.
   - Use bootstrap significance testing for fairness-gap deltas and key performance deltas.

6. **Decision and reporting**
   - Keep a model only if it improves fairness with acceptable utility tradeoff.
   - Publish model card style summary: dataset scope, subgroup behavior, limits, monitoring plan.
   - Archive config, metrics tables, plots, and commit hash.

## 12. Mitigation options by dataset profile
Choose mitigation methods by the structure of each sample CSV/dataset:

- **Binary tabular clinical outcome datasets (most common in this repo):**
  - *Pre-processing:* reweighing, stratified resampling, class/group balancing.
  - *In-processing:* Exponentiated Gradient / constrained optimization on equalized odds or demographic parity.
  - *Post-processing:* group-specific threshold optimization (equalized odds / opportunity oriented).

- **Strong class imbalance + small minority groups:**
  - Prefer reweighing + conservative thresholding first.
  - Add minimum group support rules before publishing fairness claims.
  - Use wider bootstrap intervals and highlight uncertainty.

- **High-dimensional sparse one-hot datasets:**
  - Prefer linear models with regularization for interpretability.
  - Apply reweighing before fitting; avoid aggressive over/under-sampling that distorts sparse structure.

- **Temporal or distribution-shifting datasets:**
  - Split by time (train past, test future), then run the same fairness audit.
  - Track fairness drift over time windows; re-tune thresholds per monitoring cycle.

- **Multi-site/hospital merged CSVs:**
  - Add site/hospital as audit attribute and as possible confounder.
  - Report fairness both within-site and pooled across sites.

## 13. Minimum governance controls for “trustworthy” use
- Require review of data provenance and label definition before training.
- Enforce performance guardrails (e.g., max tolerated drop in balanced accuracy).
- Enforce fairness guardrails (e.g., target upper bound for combined FPR/FNR gaps).
- Keep human-in-the-loop review for clinical deployment decisions.
