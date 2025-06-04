from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def build_faiss_index(segments, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Builds a FAISS index for efficient semantic similarity search over a list of text segments.

    Args:
        segments (List[str]): A list of preprocessed text segments to index.
        model_name (str, optional): Name of the SentenceTransformer model to use. Defaults to 'all-MiniLM-L6-v2'.

    Returns:
        Tuple:
            - faiss.IndexFlatIP: The FAISS index built on the segment embeddings.
            - np.ndarray: The array of normalized segment embeddings.
            - List[str]: The original list of input segments.
            - SentenceTransformer: The loaded SentenceTransformer model instance.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(segments, convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings, segments, model


def search_all_above_threshold(query, index, segments, model, threshold=0.55):
    """
    Checks whether a given query is semantically similar to any indexed segment above a threshold.

    Args:
        query (str): The text query to compare.
        index (faiss.IndexFlatIP): The FAISS index containing segment embeddings.
        segments (List[str]): The original list of segments (used for indexing).
        model (SentenceTransformer): The model used to encode the query.
        threshold (float, optional): Similarity threshold for considering a match. Defaults to 0.55.

    Returns:
        bool: True if the query matches at least one segment with similarity >= threshold, otherwise False.
    """
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)

    k = len(segments)
    similarities, indices = index.search(query_embedding, k)

    for i, score in zip(indices[0], similarities[0]):
        if score >= threshold:
            return True

    return False
