# Team Reimplementation Plan

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
- school-LLM explanation prompt constraints

Mengxi Zhang:
- `evaluate.py`
- `predict.py`
- `web_app.py`
- web UI
- shared `RumorPredictionService`
- adversarial/suspicious input checks
- adversarial stability evaluation plan
- tests
- Makefile
- `README.md`
- `EXPERIENTS.md`

## Experiments to Rerun in the Formal Repository

- Default training on `train.csv`.
- Validation reporting on `val.csv`.
- Single-text evidence-aware prediction with model evidence and LLM evidence.
- Web prediction with the same output contract as `predict.py`.
- Threshold and fusion metadata verification.
- Robustness checks for suspicious or perturbed input.
- Any ablation or extended data experiments the team chooses to report.

Do not copy validation artifacts or metrics from this staging repository into the final repository without rerunning them.
