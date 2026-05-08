"""
Information Retrieval utilities using NLTK for tokenization and stop‑word removal.
Provides a single class `NltkRanker` that accepts a list of documents, a query and
`top_n` and returns the `top_n` most relevant documents together with the detailed
score contributions for each term.

The ranking follows the weighted KL‑Divergence formulation used in the original
`kl_ir.py` module:

    score(d) = Σ_w IDF(w) * P_q(w) * log(P_q(w) / P_d(w))

where probabilities are Laplace‑smoothed (add‑one) and lower scores indicate higher
relevance.
"""

from __future__ import annotations

import math
import nltk
from collections import Counter
from typing import List, Dict, Tuple, Any
import re

# Ensure required NLTK resources are available. ``download`` is safe to call –
# it will be a no‑op if the data is already present.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:  # pragma: no cover – executed only on fresh envs
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:  # pragma: no cover
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize


class NltkRanker:
    """Rank documents against a query using NLTK preprocessing.

    Parameters
    ----------
    docs: List[str]
        Corpus of raw document strings.
    query: str
        Raw query string.
    top_n: int, optional
        Number of top results to return (default ``5``).
    custom_stopwords: set[str] | None, optional
        Optional user‑provided stop‑word set. If ``None`` the standard English
        stop‑words from NLTK are used.
    alpha: float, optional
        Laplace smoothing coefficient (default ``1.0``).
    """

    def __init__(
        self,
        docs: List[str],
        query: str,
        top_n: int = 5,
        custom_stopwords: set[str] | None = None,
        alpha: float = 1.0,
    ) -> None:
        self.docs_raw = docs
        self.query_raw = query
        self.top_n = top_n
        self.alpha = alpha
        self.stopwords = (
            custom_stopwords
            if custom_stopwords is not None
            else set(nltk_stopwords.words("english"))
        )
        # Preprocess once – token lists for docs and query.
        self.docs_tokens = [self._preprocess(d) for d in self.docs_raw]
        self.query_tokens = self._preprocess(self.query_raw)
        # Build IDF from the document corpus (query is not part of the IDF).
        self.idf = self._compute_idf(self.docs_tokens)
        # Cache term frequencies for efficiency.
        self.docs_tf = [Counter(toks) for toks in self.docs_tokens]
        self.query_tf = Counter(self.query_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize using NLTK's ``word_tokenize`` and lower‑case the result."""
        return [tok.lower() for tok in word_tokenize(text)]
    
    def _stemming(self,tokens: list[str]) -> list[str]:
        """Apply Porter stemming to a list of tokens."""
        porter = nltk.PorterStemmer()
        return [porter.stem(tok) for tok in tokens]

    def _preprocess(self, text: str) -> List[str]:
        """Tokenize and remove stop‑words.

        Empty strings after stop‑word removal are discarded – they simply do not
        contribute to the model.
        """
        clean_text = re.sub(r'[^\w\s]', '', text)
        tokens = self._tokenize(clean_text)
        clear_word = [t for t in tokens if t not in self.stopwords ]
        
        clear_word = self._stemming(clear_word)
        return clear_word

    @staticmethod
    def _compute_idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
        """Compute IDF for each term in the corpus.

        IDF(w) = log(N / df(w)) where ``N`` is the number of documents and ``df``
        is the document frequency of ``w``.
        """
        N = len(docs_tokens)
        df: Dict[str, int] = {}
        for tokens in docs_tokens:
            unique = set(tokens)
            for term in unique:
                df[term] = df.get(term, 0) + 1
        return {term: math.log(N / freq) for term, freq in df.items()}

    def _laplace_prob(self, term: str, tf: Counter, total_tokens: int) -> float:
        """Laplace‑smoothed probability of ``term``.

        P(w|doc) = (count(w) + α) / (|V|·α + total_tokens)
        where ``|V|`` is the vocabulary size of the corpus.
        """
        vocab_size = len(self.idf)
        count = tf.get(term, 0)
        return (count + self.alpha) / (total_tokens + self.alpha * vocab_size)

    def _score_document(self, doc_index: int) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted KL‑divergence for a single document.

        Returns
        -------
        score: float
            The KL‑divergence (lower = more relevant).
        details: dict
            Per‑term contribution ``idf * P_q * log(P_q / P_d)`` for debugging.
        """
        doc_tf = self.docs_tf[doc_index]
        total_doc_tokens = sum(doc_tf.values())
        total_query_tokens = sum(self.query_tf.values())
        score = 0.0
        details: Dict[str, float] = {}
        for term, q_count in self.query_tf.items():
            idf_val = self.idf.get(term, 0.0)
            p_q = self._laplace_prob(term, self.query_tf, total_query_tokens)
            p_d = self._laplace_prob(term, doc_tf, total_doc_tokens)
            # Guard against division by zero – p_d is never zero because of smoothing.
            contribution = idf_val * p_q * math.log(p_q / p_d)
            details[term] = contribution
            score += contribution
        return score, details

    def rank(self) -> List[Dict[str, Any]]:
        """Return the top ``top_n`` documents with detailed scores.

        The return format matches the request: a list of length ``top_n`` where
        each entry contains the document index, raw score and a mapping of term
        contributions.
        """
        scores: List[Tuple[int, float, Dict[str, float]]] = []
        for i in range(len(self.docs_raw)):
            s, details = self._score_document(i)
            scores.append((i, s, details))
        # Sort by ascending score (lower is better) and slice.
        scores.sort(key=lambda x: x[1])
        top = scores[: self.top_n]
        return [
            {"doc_id": doc_idx, "score": sc, "term_contributions": contrib}
            for doc_idx, sc, contrib in top
        ]

# Example usage (can be removed in production code)
if __name__ == "__main__":
    
    docs = [
    "Maintaining a balanced diet rich in fiber and vitamins is essential for digestive health.",
    "Regular cardiovascular exercise can significantly reduce the risk of heart disease.",
    "Mental health is just as important as physical health for overall well-being.",
    "Getting at least eight hours of sleep helps the body repair tissues and boost the immune system.",
    "Chronic stress can lead to high blood pressure and other long-term health complications.",
    "Drinking enough water throughout the day is crucial for maintaining proper organ function.",
    "Vaccinations are a key public health tool in preventing the spread of infectious diseases.",
    "Practicing mindfulness and meditation can help manage anxiety and improve mental clarity.",
    "Sugar consumption should be limited to prevent the onset of type 2 diabetes and obesity.",
    "Early detection through regular medical check-ups can improve the success rate of cancer treatments."
]
    
    
    query = "how to prevent diseases and improve the body's immune system through lifestyle"
    print(f"query : {query}")
    
    ranker = NltkRanker(docs, query, top_n=3)
    
    for result in ranker.rank():
        print(result)
        print("Document:", docs[result["doc_id"]])
        # print(docs[result["doc_id"]])
