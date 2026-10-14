from .serper_retriever import SerperEvidenceRetriever, RetrievalError  # noqa: F401

retriever_map = {
    "serper": SerperEvidenceRetriever,
}


def retriever_mapper(retriever_name: str):
    if retriever_name == "google":
        # Lazy import: the Google scraper needs heavy optional packages (spacy, sentence-transformers, torch).
        from .google_retriever import GoogleEvidenceRetriever

        return GoogleEvidenceRetriever
    if retriever_name not in retriever_map:
        raise NotImplementedError(f"Retriever {retriever_name} not found!")
    return retriever_map[retriever_name]
