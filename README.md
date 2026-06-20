# rumor-detection-model-for-AI-class

A simple model estabiled by zsf&amp;lzh&amp;zmx, just a sample designed by 3 college students from SJTU. Maybe we'll improve it in the long run.

Our goal is to build a rumor detector for short social-media style claims.

The final system is designed to combine text classification with retrieval evidence and lightweight claim-structure signals.

This repository contains one final, reproducible rumor detection pipeline for
tweet-level binary classification. The retained pipeline is evidence-aware:

1. normalize the input text;
2. score it with a tuned TF-IDF ensemble;
3. retrieve similar training cases as evidence;
4. extract lightweight claim-structure features;
5. fuse model, retrieval, and structure signals with tuned weights;
6. return the final label, probability, confidence, evidence, and explanation.

Label `1` means `rumor`; label `0` means `non-rumor`.

## Installation

```bash
python -m pip install -r requirements.txt
```

or:

```bash
make install
```

The final pipeline only requires Python and `numpy`.

## Data Format

`train.csv` and `val.csv` must contain these columns:

```text
id,text,label,event
```

`label` must be `0` or `1`. `event` is used for stratified internal splitting
and per-event diagnostics.

## Train

```bash
python train.py --train train.csv --val val.csv --model models/ensemble.pkl --metrics outputs/metrics.json
```

or:

```bash
make train
```

Training uses `train.csv` only for model fitting and internal tuning. It creates
a stratified internal dev split from `train.csv`, tunes the ensemble threshold
and evidence-fusion parameters on that internal dev split, then reports
validation metrics on `val.csv`. `val.csv` is not used for tuning.

Outputs:

```text
models/ensemble.pkl
outputs/metrics.json
```

The model artifact stores the selected sub-models, ensemble weights, base
threshold, evidence weights, evidence top-k, retrieval similarity gate, and
evidence-aware threshold.

Evidence weights are selected on the internal dev split. If a signal does not
improve the dev objective, its tuned weight can be `0.0`; the prediction JSON
always exposes the effective `decision_factors` used for the final probability.

## Evaluate

```bash
python evaluate.py --model models/ensemble.pkl --data val.csv --train train.csv --out-dir outputs
```

or:

```bash
make evaluate
```

Evaluation always uses the final evidence-aware pipeline and the saved
evidence-aware threshold.

Outputs:

```text
outputs/evaluation.json
outputs/predictions.csv
outputs/correct_cases.csv
outputs/wrong_cases.csv
outputs/explain_cases.json
```

`evaluation.json` contains overall and per-event metrics. `predictions.csv`
contains one row per evaluated sample. `explain_cases.json` keeps structured
evidence for selected correct and wrong cases.

## Predict

```bash
python predict.py --model models/ensemble.pkl --train train.csv --text "Swiss museum confirms it will take on #Gurlitt collection"
```

or:

```bash
make predict TEXT="Swiss museum confirms it will take on #Gurlitt collection"
```

Prediction returns JSON with:

```text
label
label_name
prob_rumor
confidence
threshold
baseline_label
baseline_prob_rumor
evidence
explanation
```

The `evidence` object includes keyword contributions, retrieved training cases,
retrieval statistics, claim-structure features, and fusion decision factors.

## Tests

```bash
python -m unittest discover -s tests
```

or:

```bash
make test
```

## Project Structure

```text
.
|-- train.py                 # train and tune the final pipeline
|-- evaluate.py              # evaluate the final pipeline
|-- predict.py               # single-text prediction with evidence
|-- Makefile                 # main commands only
|-- README.md
|-- EXPERIMENTS.md
|-- requirements.txt
|-- train.csv
|-- val.csv
|-- tests/
`-- src/
    |-- text_model.py        # text normalization, TF-IDF model, metrics
    |-- ensemble_model.py    # TF-IDF ensemble
    |-- defense.py           # shared input sanitization
    |-- retriever.py         # training-case retrieval
    |-- evidence.py          # retrieval evidence features
    |-- claim_structure.py   # lightweight claim-structure features
    `-- evidence_pipeline.py # final evidence-aware inference pipeline
```

Generated artifacts are intentionally not part of the source workflow:

```text
models/
outputs/
```