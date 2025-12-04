# src/bow.py

# this file implements a Bag-of-Words (BoW) model
# it can build a vocabulary from a list of tokens
# and vectorize a list of tokenized documents into a sparse matrix

from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np

class BoW:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def _word_to_index(self, sorted_tokens):
        return {token: i for i, (token, _) in enumerate(sorted_tokens)}

    def _index_to_word(self, sorted_tokens):
        return [token for token, _ in sorted_tokens]

    def build_vocabulary(self, sorted_tokens):
        self.idx2word = self._index_to_word(sorted_tokens)
        self.word2idx = self._word_to_index(sorted_tokens)
        return self.word2idx, self.idx2word

    def vectorize_list(self, token_list):
        """
        creates a SPARSE BoW-Matrix (csr_matrix) from a list of tokenized documents.
        token_list: list of tweets, every tweet = list of tokens.
        returns: csr_matrix (num_docs x vocab_size)
        """
        rows = []
        cols = []
        data = []

        # count word occurrences per document
        for row_idx, tokens in enumerate(token_list):
            counts = Counter(tokens)
            # fill the sparse matrix data
            for token, count in counts.items():
                idx = self.word2idx.get(token)
                if idx is not None:
                    rows.append(row_idx)
                    cols.append(idx)
                    data.append(count)

        num_docs = len(token_list)
        vocab_size = len(self.word2idx)

        # convert to Sparse-Matrix
        X = csr_matrix(
            (np.array(data, dtype=np.float32),
             (np.array(rows), np.array(cols))),
            shape=(num_docs, vocab_size),
        )
        return X