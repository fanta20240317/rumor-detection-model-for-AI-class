"""Evidence feature builders used before the final rumor decision."""


def build_retrieval_evidence_features(retriever, text, top_k=5, exclude_exact=True):
    """Retrieve top-k cases and summarize them as decision features.
    
    Unlike the older explanation-only path, this function does not filter by a
    predicted label. It first observes the nearest training cases, then derives
    label ratios and similarity statistics that can participate in the final
    decision.
    """
    if retriever is None:
        return {
            "retrieved_cases": [],
            "retrieval_statistics": empty_retrieval_statistics(top_k),
        }

    cases = retriever.search(text, label=None, top_k=top_k, exclude_exact=exclude_exact)
    stats = summarize_retrieved_cases(cases, top_k=top_k)
    return {"retrieved_cases": cases, "retrieval_statistics": stats}


def empty_retrieval_statistics(top_k=5):
    return {
        "top_k": top_k,
        "retrieved_count": 0,
        "retrieved_rumor_ratio": 0.5,
        "retrieved_nonrumor_ratio": 0.5,
        "max_similarity": 0.0,
        "mean_similarity": 0.0,
        "weighted_rumor_score": 0.5,
        "retrieval_confidence": 0.0,
        "retrieval_margin": 0.0,
        "label_consistency": 0.0,
    }


def summarize_retrieved_cases(cases, top_k=5):
    if not cases:
        return empty_retrieval_statistics(top_k)

    count = len(cases)
    rumor_count = sum(1 for item in cases if item["label"] == 1)
    nonrumor_count = count - rumor_count
    scores = [max(float(item.get("score", 0.0)), 0.0) for item in cases]
    total_score = sum(scores)

    if total_score > 0:
        weighted_rumor_score = sum(
            score * item["label"] for score, item in zip(scores, cases)
        ) / total_score
    else:
        weighted_rumor_score = rumor_count / count

    rumor_ratio = rumor_count / count
    nonrumor_ratio = nonrumor_count / count
    label_consistency = max(rumor_ratio, nonrumor_ratio)
    margin = abs(weighted_rumor_score - 0.5) * 2.0
    max_similarity = max(scores)
    mean_similarity = total_score / count

    # Scores in this sparse TF-IDF retriever are often modest; 0.12 already
    # indicates useful lexical overlap in this dataset.
    similarity_strength = min(max_similarity / 0.12, 1.0)
    coverage = min(count / max(top_k, 1), 1.0)
    retrieval_confidence = similarity_strength * coverage * max(margin, 0.05)

    return {
        "top_k": top_k,
        "retrieved_count": count,
        "retrieved_rumor_ratio": round(rumor_ratio, 4),
        "retrieved_nonrumor_ratio": round(nonrumor_ratio, 4),
        "max_similarity": round(max_similarity, 4),
        "mean_similarity": round(mean_similarity, 4),
        "weighted_rumor_score": round(weighted_rumor_score, 4),
        "retrieval_confidence": round(min(retrieval_confidence, 1.0), 4),
        "retrieval_margin": round(margin, 4),
        "label_consistency": round(label_consistency, 4),
    }


def compact_keyword_evidence(base_result):
    return [
        {"term": term, "source": "tfidf_contribution"}
        for term in base_result.get("evidence_terms", [])
    ]