import uuid

import chromadb
from chromadb.utils import embedding_functions

_CROSS_ENCODER = None
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(_CROSS_ENCODER_MODEL)
    return _CROSS_ENCODER


class TADataRetriever:
    """ChromaDB-backed vector store for rubric and reference-solution retrieval."""

    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="grading_context",
            embedding_function=self.embed_fn,
        )

    def clear_index(self):
        """Removes all documents so a new grading run does not stack duplicate chunks."""
        batch = self.collection.get(include=[])
        ids = batch.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)

    def add_to_index(self, text, metadata):
        """Chunk text, embed, and add documents to the collection."""
        chunks = self._chunk_text(text, size=500, overlap=100)
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [metadata for _ in chunks]
        
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

    def _chunk_text(self, text, size, overlap):
        """Fixed-size overlapping windows over character spans."""
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunks.append(text[i:i + size])
        return chunks

    def retrieve_relevant_context(self, query, top_k=3, candidate_k=24):
        """Retrieve Chroma candidates by embedding similarity, then rerank with a cross-encoder."""
        n_docs = self.collection.count()
        if n_docs == 0:
            return ""
        fetch_k = min(n_docs, max(top_k, candidate_k))
        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
        )
        docs = results["documents"][0]
        if not docs:
            return ""

        ce = _get_cross_encoder()
        pairs = [[query, d] for d in docs]
        scores = ce.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        top_docs = [docs[i] for i in order[:top_k]]
        return "\n\n".join(top_docs)