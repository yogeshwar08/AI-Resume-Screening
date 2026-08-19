try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. "
        "Install it with: pip install sentence-transformers"
    )

from sklearn.metrics.pairwise import cosine_similarity


# Lightweight and strong general-purpose model
MODEL_NAME = "all-MiniLM-L6-v2"


model = SentenceTransformer(MODEL_NAME)


def calculate_semantic_scores(
    job_description: str,
    resumes: list[str]
) -> list[float]:
    """
    Calculate semantic similarity between
    job description and multiple resumes.

    Returns scores from 0 to 100.
    """

    # Encode job description
    job_embedding = model.encode(
        [job_description],
        normalize_embeddings=True
    )

    # Encode resumes
    resume_embeddings = model.encode(
        resumes,
        normalize_embeddings=True
    )

    # Cosine similarity
    similarities = cosine_similarity(
        job_embedding,
        resume_embeddings
    )[0]

    # Convert similarity to 0-100
    scores = []

    for similarity in similarities:

        # Cosine similarity can theoretically be -1 to 1
        score = ((float(similarity) + 1) / 2) * 100

        scores.append(
            round(score, 2)
        )

    return scores