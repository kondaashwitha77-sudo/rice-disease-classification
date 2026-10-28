# MLP2 Git Workflow — Complete Branch Commands

## Pipeline → Branch Mapping

```
main
└── develop
    ├── feature/01-dataset-collection
    ├── feature/02-dataset-splitting
    ├── feature/03-image-preprocessing
    ├── feature/04-feature-extraction
    ├── feature/05-feature-preparation
    ├── feature/06-model-training
    ├── feature/07-validation
    ├── feature/08-final-test-evaluation
    ├── feature/09-error-analysis
    ├── feature/10-deployment
    ├── feature/11-learning
    ├── feature/12-retraining
    └── feature/13-reporting-documentation
```

---

## Step 0 — Initialize Repository

```bash
cd "c:\Users\Ashwitha\Downloads\MLP2\MLP2"
git init
git add .
git commit -m "chore: initial project setup"
```

---

## Step 1 — Create `main` and `develop` base branches

```bash
# Rename default branch to main (if not already)
git branch -M main

# Create develop branch from main
git checkout -b develop
git push -u origin develop
```

---

## Step 2 — Create All Feature Branches from `develop`

Run these commands one after another:

```bash
# 01 — Dataset Collection
git checkout develop
git checkout -b feature/01-dataset-collection

# 02 — Dataset Splitting (70% train / 15% val / 15% test, stratified)
git checkout develop
git checkout -b feature/02-dataset-splitting

# 03 — Image Preprocessing (resize, normalize, format standardize)
git checkout develop
git checkout -b feature/03-image-preprocessing

# 04 — Feature Extraction (MCLBP + Color Correlogram)
git checkout develop
git checkout -b feature/04-feature-extraction

# 05 — Feature Preparation (scaling, PCA, fusion comparison)
git checkout develop
git checkout -b feature/05-feature-preparation

# 06 — Model Training (classical ML: SVM, RF, ExtraTrees, etc.)
git checkout develop
git checkout -b feature/06-model-training

# 07 — Validation (tune on val set, cross-validation)
git checkout develop
git checkout -b feature/07-validation

# 08 — Final Test Evaluation (run on held-out test set ONCE)
git checkout develop
git checkout -b feature/08-final-test-evaluation

# 09 — Error Analysis (confusion matrix, misclassifications)
git checkout develop
git checkout -b feature/09-error-analysis

# 10 — Deployment (Streamlit app, predict.py, saved models)
git checkout develop
git checkout -b feature/10-deployment

# 11 — Learning (document lessons learned)
git checkout develop
git checkout -b feature/11-learning

# 12 — Retraining (retrain with new data or better hyperparams)
git checkout develop
git checkout -b feature/12-retraining

# 13 — Reporting & Documentation (README, reports/, final paper)
git checkout develop
git checkout -b feature/13-reporting-documentation
```

---

## Step 3 — Typical Work Cycle Per Branch

```bash
# Switch to the branch you're working on
git checkout feature/04-feature-extraction

# ... make changes to files ...

# Stage and commit
git add .
git commit -m "feat(04): add MCLBP + Color Correlogram extraction"

# Push branch to remote
git push -u origin feature/04-feature-extraction

# When done — merge back to develop
git checkout develop
git merge --no-ff feature/04-feature-extraction -m "merge: feature/04-feature-extraction into develop"
git push origin develop
```

---

## Step 4 — Release to `main` (after all pipeline steps are complete)

```bash
git checkout main
git merge --no-ff develop -m "release: MLP2 pipeline v1.0 complete"
git tag -a v1.0 -m "MLP2 v1.0 — full pipeline complete"
git push origin main --tags
```

---

## Branch Protection Rules (Recommended)

| Branch       | Rule                                      |
|--------------|-------------------------------------------|
| `main`       | No direct push; merge via PR only         |
| `develop`    | No direct push; merge via PR from feature |
| `feature/*`  | Direct push allowed; delete after merge   |

---

## Quick Reference — All Branches at a Glance

| # | Branch Name                          | Pipeline Step              |
|---|--------------------------------------|----------------------------|
| – | `main`                               | Production-ready code      |
| – | `develop`                            | Integration branch         |
| 1 | `feature/01-dataset-collection`      | Dataset collection         |
| 2 | `feature/02-dataset-splitting`       | 70/15/15 stratified split  |
| 3 | `feature/03-image-preprocessing`     | Resize, normalize, format  |
| 4 | `feature/04-feature-extraction`      | MCLBP + Color Correlogram  |
| 5 | `feature/05-feature-preparation`     | Scale, PCA, fusion         |
| 6 | `feature/06-model-training`          | Train ML classifiers       |
| 7 | `feature/07-validation`              | Tune on val set            |
| 8 | `feature/08-final-test-evaluation`   | Final held-out test        |
| 9 | `feature/09-error-analysis`          | Confusion matrix, errors   |
|10 | `feature/10-deployment`              | Streamlit app + predict    |
|11 | `feature/11-learning`                | Lessons learned            |
|12 | `feature/12-retraining`              | Retrain with new data      |
|13 | `feature/13-reporting-documentation` | README, reports, paper     |
