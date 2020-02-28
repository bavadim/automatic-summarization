#!/usr/bin/env python3

from typing import List, Tuple
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from nltk.tokenize import sent_tokenize


class BaseSummarizer(object):
    ROUND_DIGITS = 5

    def __text2sentences__(self, text: str) -> List[str]:
        raise NotImplementedError

    def __embeddings__(self, sentences: List[str]) -> tf.Tensor:
        raise NotImplementedError

    def __sim_mat__(self, vec: tf.Tensor) -> tf.Tensor:
        normalize = tf.math.l2_normalize(vec, 1)
        cosine = tf.linalg.matmul(normalize, normalize, transpose_b=True)
        rounded = tf.math.round(cosine * 10 ** BaseSummarizer.ROUND_DIGITS) / 10 ** BaseSummarizer.ROUND_DIGITS
        return rounded

    @staticmethod
    def __ranks__(sent_sim_mat: tf.Tensor) -> tf.Tensor:
        eig_val, eig_vec = tf.linalg.eigh(sent_sim_mat)
        best_vector_idx = tf.math.argmax(eig_val)
        return eig_vec[best_vector_idx]

    @staticmethod
    def __z_score__(vec: tf.Tensor) -> tf.Tensor:
       return (vec - tf.math.reduce_min(vec)) / (tf.math.reduce_max(vec) - tf.math.reduce_min(vec))

    def scored_sentences(self, text: str) -> List[Tuple[str, float]]:
        sents = self.__text2sentences__(text)
        if not sents:
            return []
        sim_mat = self.__sim_mat__(self.__embeddings__(sents))
        ranks = BaseSummarizer.__z_score__(BaseSummarizer.__ranks__(sim_mat))
        return list(zip(sents, ranks.numpy()))

    def the_most_important(self, text, k=1):
        return [ p[0] for p in sorted(self.scored_sentences(text), key=lambda p: p[1], reverse=True)[:k] ]


class USETextRank(BaseSummarizer):
    __embed__ = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    def __embeddings__(self, sentences: List[str]) -> tf.Tensor:
        return self.__embed__(sentences)

    def __text2sentences__(self, text: str) -> List[str]:
        return sent_tokenize(text)


if __name__ == '__main__':
    summarizer = USETextRank()
    buffer = ''
    for line in sys.stdin:
        if line == '\n' and buffer != '':
            print(summarizer.scored_sentences(buffer))
            buffer = ''
        else:
            buffer += line
