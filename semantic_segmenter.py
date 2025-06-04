from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSegmenter:
    """
    A class for semantic text segmentation based on cosine similarity between sentences.
    """

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the sentence-transformers model for generating embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def split_into_sentences(self, text):
        """
        Splits the input text into individual sentences.
        """
        return nltk.sent_tokenize(text)

    def embed_sentences(self, sentences):
        """
        Computes embeddings for each sentence.
        """
        return self.model.encode(sentences)

    def get_similarity_scores(self, embeddings):
        """
        Computes cosine similarity scores between consecutive sentence embeddings.
        """
        return [
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ]

    def segment_text(self, sentences, embeddings, threshold=0.7):
        """
        Segments the text based on a similarity threshold.
        """
        similarity_scores = self.get_similarity_scores(embeddings)
        segments = []
        current_segment = [sentences[0]]

        for i in range(1, len(sentences)):
            if similarity_scores[i - 1] < threshold:
                segments.append(' '.join(current_segment))
                current_segment = [sentences[i]]
            else:
                current_segment.append(sentences[i])

        if current_segment:
            segments.append(' '.join(current_segment))

        return segments

    def segment(self, text, threshold=0.55):
        """
        Main method: segments the input text into semantically coherent parts.
        """
        sentences = self.split_into_sentences(text)
        embeddings = self.embed_sentences(sentences)
        return self.segment_text(sentences, embeddings, threshold)