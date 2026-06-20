# Evidence Pipeline Design

Evaluation loads the saved evidence-aware pipeline, scores `val.csv`, and reports validation metrics. `val.csv` is not described as an independent test set.

Metrics use the saved evidence_threshold, matching the labels exported in prediction rows.

Evaluation exports aggregate metrics, per-event metrics, predictions, and case explanations under outputs/.

Evidence-first prediction augments the base TF-IDF probability with retrieved neighbors and lightweight claim-structure features. All feature directions are aligned so higher scores indicate stronger support for `P(rumor)`.
The retrieval index is built from 	rain.csv so evaluation and prediction can cite comparable training claims without using validation labels as evidence.

Retriever output keeps the top-k similar cases with item id, label, event, text, and similarity for downstream evidence features.

Evidence features include retrieved rumor/non-rumor ratios so retrieved neighbors can influence the final probability directionally.

Similarity statistics summarize the strength and consistency of the retrieved evidence set.

Weighted rumor score and retrieval margin are converted into compact features for evidence-aware fusion.

Claim structure features capture URL, mention, hashtag, and punctuation patterns that often affect rumor risk.

The structure module also captures uncertainty and source-word cues without introducing external facts.

The lightweight structure risk score remains a bounded auxiliary signal, not a separate model family.

Fusion combines base model probability, retrieval evidence score, and structure score into a final evidence-aware P(rumor).

Base probability remains the dominant signal while retrieval and structure act as calibrated supporting signals.

Conservative fusion weights reduce the chance that sparse or weak evidence overwhelms the text classifier.

The saved evidence_threshold is the label threshold used by evaluation and prediction.

All fusion scores are aligned with 1 = rumor, so inal_prob always means P(rumor).

RumorDetectionPipeline connects ensemble probabilities, retriever results, structure features, fusion, and explanation fields in one prediction path.

Each prediction returns structured evidence so downstream JSON and reports can be traced back to the same model run.

Decision factors record the contribution of base, retrieval, and structure signals for explanation and debugging.

Explanation is generated from the same evidence bundle used by the final prediction.

Explanations are restricted to structured evidence, keyword contributions, and retrieved cases; no outside facts are introduced.

Keyword summaries help reviewers inspect which normalized text features supported the base classifier branch.

Retrieved case summaries expose top evidence examples with similarity and labels for transparency.