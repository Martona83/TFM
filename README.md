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
