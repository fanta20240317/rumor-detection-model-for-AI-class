# Evidence Pipeline Design

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