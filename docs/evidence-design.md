# Evidence Pipeline Design

Evidence-first prediction augments the base TF-IDF probability with retrieved neighbors and lightweight claim-structure features. All feature directions are aligned so higher scores indicate stronger support for `P(rumor)`.
The retrieval index is built from 	rain.csv so evaluation and prediction can cite comparable training claims without using validation labels as evidence.