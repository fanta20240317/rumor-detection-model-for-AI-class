# Evidence Pipeline Design

Evidence-first prediction augments the base TF-IDF probability with retrieved neighbors and lightweight claim-structure features. All feature directions are aligned so higher scores indicate stronger support for `P(rumor)`.
The retrieval index is built from 	rain.csv so evaluation and prediction can cite comparable training claims without using validation labels as evidence.

Retriever output keeps the top-k similar cases with item id, label, event, text, and similarity for downstream evidence features.