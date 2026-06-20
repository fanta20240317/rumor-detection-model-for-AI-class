# Prediction Design

Prediction loads the same `RumorDetectionPipeline` used by evaluation and emits evidence-aware JSON for a single input text.

The JSON includes label, probability, confidence, threshold, evidence, decision factors, and explanation.

Missing model artifacts or evidence corpus paths should fail clearly instead of silently falling back to another model.

make predict uses the default model and train evidence paths with an overridable TEXT variable.