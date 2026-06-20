# Evaluation Design

Evaluation loads the saved evidence-aware pipeline, scores `val.csv`, and reports validation metrics. `val.csv` is not described as an independent test set.
Metrics use the saved evidence_threshold, matching the labels exported in prediction rows.

Evaluation exports aggregate metrics, per-event metrics, predictions, and case explanations under outputs/.

Evaluation does not call the school LLM. LLM evidence is generated only by
terminal or web prediction after the local model has produced its result, so
validation metrics remain deterministic and independent of external API state.

Reported metrics are validation performance on val.csv; the project has no independent test.csv.

Final tests are consolidated in tests/test_project.py so command and pipeline contracts are checked in one place.

README and EXPERIENTS must describe val.csv as validation data only; no independent test accuracy is claimed.
