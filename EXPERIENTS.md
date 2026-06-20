# Experiments

This file documents the retained final experiment protocol. Historical
lightweight, transformer-fusion, adversarial, ablation, and web-demo workflows
were removed from the main project to keep one reproducible pipeline.

## Final Pipeline

The final model is an evidence-aware TF-IDF ensemble:

```text
normalized text
-> TF-IDF ensemble probability
-> top-k retrieval evidence from the training set
-> claim-structure features
-> tuned evidence-aware fusion
-> retrieval accuracy guard
-> final thresholded label and explanation
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

Evaluation and prediction both use the same `RumorDetectionPipeline` and the
same saved evidence-aware threshold.
Evidence fusion weights are tuned, so a retrieval or structure stream can have
weight `0.0` when the internal dev split does not benefit from it; the emitted
`evidence.decision_factors` show the effective weights used for each prediction.
The saved retrieval accuracy guard can lower high-confidence non-rumor cases or
rescue borderline rumor cases before thresholding.

## Reproduction

```bash
python -m pip install -r requirements.txt
python train.py --train train.csv --val val.csv --model models/main_fusion.pkl --metrics outputs/metrics.json
python evaluate.py --model models/main_fusion.pkl --data val.csv --train train.csv --out-dir outputs
python predict.py --model models/main_fusion.pkl --train train.csv --text "Swiss museum confirms it will take on #Gurlitt collection"
python -m unittest discover -s tests
```

The equivalent Make targets are:

```bash
make install
make train
make evaluate
make predict TEXT="Swiss museum confirms it will take on #Gurlitt collection"
make test
```

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

This keeps evaluation, prediction, and explanation aligned to the same final
pipeline.
