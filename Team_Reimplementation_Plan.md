# Team Reimplementation Plan

This staging repository is a reference construction plan. The `plan(Person A/B/C)` labels in commit messages are suggested responsibilities for a future formal repository. They do not represent authorship in this staging repository.

## Responsibilities

Shifei Zheng:
- project skeleton
- data loading
- text normalization
- TF-IDF baseline
- TF-IDF dual ensemble
- `train.py`

Zihan Li:
- retriever
- retrieval evidence features
- claim structure features
- evidence-aware fusion
- `RumorDetectionPipeline`
- evidence-first explanation

Mengxi Zhang:
- `evaluate.py`
- `predict.py`
- adversarial/suspicious input checks
- adversarial stability evaluation plan
- tests
- Makefile
- `README.md`
- `EXPERIMENTS.md`

## Experiments to Rerun in the Formal Repository

- Default training on `train.csv`.
- Validation reporting on `val.csv`.
- Single-text evidence-aware prediction.
- Threshold and fusion metadata verification.
- Robustness checks for suspicious or perturbed input.
- Any ablation or extended data experiments the team chooses to report.

Do not copy validation artifacts or metrics from this staging repository into the final repository without rerunning them.