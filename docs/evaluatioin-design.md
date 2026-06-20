# Evaluation Design

Evaluation loads the saved evidence-aware pipeline, scores `val.csv`, and reports validation metrics. `val.csv` is not described as an independent test set.
Metrics use the saved evidence_threshold, matching the labels exported in prediction rows.

Evaluation exports aggregate metrics, per-event metrics, predictions, and case explanations under outputs/.

Reported metrics are validation performance on al.csv; the project has no independent 	est.csv.

Final tests are consolidated in 	ests/test_project.py so command and pipeline contracts are checked in one place.