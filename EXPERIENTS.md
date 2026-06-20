# Experiments

This file documents the retained final experiment protocol. Historical
lightweight, transformer-fusion, adversarial, and ablation workflows were
removed from the main project to keep one reproducible pipeline. The current
web UI is an interface for the retained final prediction service, not a
separate experimental model.

## Final Pipeline

The final model is an evidence-aware TF-IDF ensemble:

```text
normalized text
-> TF-IDF ensemble probability
-> top-k retrieval evidence from the training set
-> claim-structure features
-> tuned evidence-aware fusion
-> retrieval accuracy guard
-> final thresholded label and local evidence explanation
-> optional school-LLM evidence explanation
```

The saved model artifact contains:

- selected TF-IDF sub-models;
- ensemble weights;
- base ensemble threshold;
- evidence fusion weights;
- evidence top-k;
- retrieval similarity gate;
- evidence-aware threshold;
- retrieval accuracy guard settings.

Evaluation uses the saved `RumorDetectionPipeline` and evidence-aware
threshold. Terminal and web prediction use `RumorPredictionService`, which wraps
the same pipeline and can attach school-LLM evidence after the local decision.
Evidence fusion weights are tuned, so a retrieval or structure stream can have
weight `0.0` when the internal dev split does not benefit from it; the emitted
`evidence.decision_factors` show the effective weights used for each prediction.
The saved retrieval accuracy guard can lower high-confidence non-rumor cases or
rescue borderline rumor cases before thresholding.
LLM evidence is not used for training, thresholding, or metrics; it only
rewrites the already emitted model evidence into a Chinese explanation.
The LLM prompt includes the original tweet, predicted label, confidence, model
evidence terms, and RAG-style similar samples. The local reproducible model
remains the only source of classification labels.

## Reproduction

```bash
python -m pip install -r requirements.txt
python train.py --train train.csv --val val.csv --model models/main_fusion.pkl --metrics outputs/metrics.json
python evaluate.py --model models/main_fusion.pkl --data val.csv --train train.csv --out-dir outputs
python predict.py --model models/main_fusion.pkl --train train.csv --text "Swiss museum confirms it will take on #Gurlitt collection"
python web_app.py --model models/main_fusion.pkl --train train.csv
python -m unittest discover -s tests
```

The equivalent Make targets are:

```bash
make install
make train
make evaluate
make predict TEXT="Swiss museum confirms it will take on #Gurlitt collection"
make web
make test
```

Prediction attempts to include `llm_evidence` by default. Configure the school
API with `SCHOOL_LLM_API_KEY`, `SCHOOL_LLM_BASE_URL` or `SCHOOL_LLM_API_URL`,
and `SCHOOL_LLM_MODEL`. Use `python predict.py --no-llm ...` to disable LLM
evidence for a single terminal run.

## Data Split Protocol

`train.py` performs a stratified internal split of `train.csv` by
`event + label`.

- fit split: trains candidate TF-IDF sub-models;
- internal dev split: selects ensemble members, ensemble weights, threshold,
  evidence top-k, fusion weights, and retrieval similarity gate;
- `val.csv`: validation evaluation only.

The internal dev evidence retriever is built only from the fit split, so dev
samples do not enter their own evidence corpus during tuning. The final
evaluation and prediction commands use `train.csv` as the evidence corpus,
which matches deployment behavior. After internal selection, the final selected
TF-IDF branches are refit on the full `train.csv`.
The retrieval accuracy guard settings are fixed in the final pipeline and saved
inside the model artifact.

## Metrics

The main reported metrics are:

- accuracy;
- precision;
- recall;
- F1;
- confusion matrix;
- threshold used for the final evidence-aware decision.

Metrics in `outputs/evaluation.json` are validation metrics computed from the
final evidence-aware probabilities and the saved evidence-aware threshold. They
do not report a separate baseline model result.

## Output Interpretation

Prediction JSON includes both final and baseline fields:

- `prob_rumor`: final evidence-aware probability;
- `threshold`: saved final evidence-aware threshold;
- `baseline_prob_rumor`: TF-IDF ensemble probability before evidence fusion;
- `evidence.retrieval_statistics`: statistics from retrieved training cases;
- `evidence.claim_structure_features`: lightweight structure signals;
- `evidence.decision_factors`: effective fusion factors used for the final
  probability;
- `evidence.probability_guard`: whether the retrieval accuracy guard adjusted
  the final probability;
- `explanation`: text generated from the same evidence used by the final
  decision.
- `llm_evidence`: school-LLM explanation generated from the final model output
  and evidence, or an unavailable status if the API is not configured.

This keeps evaluation, prediction, local explanation, and LLM explanation
aligned to the same final model evidence.
