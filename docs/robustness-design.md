# Robustness Design

The current project includes input sanitization for zero-width characters, homoglyphs, repeated characters, and whitespace perturbations through `src/defense.py`.
Sanitization runs before feature extraction, so obfuscation handling is integrated into the evidence-aware prediction path.

Unit tests in the final project cover suspicious character normalization and label stability under common perturbations.

No independent adversarial benchmark results are staged here; final submissions must rerun robustness checks in the formal repository.

Perturbation categories documented for formal replication: punctuation, casing, URL, whitespace, zero-width, and homoglyph edits.

Recommended robustness reporting: clean accuracy, attacked accuracy, accuracy drop, and prediction consistency.

Use a small --limit style option if a formal adversarial evaluation script is added during reimplementation for quick checks.

Robustness usage is documented as a reimplementation expectation rather than a fabricated completed experiment.

The formal repository should rerun ablation and robustness checks and record fresh results from that run.