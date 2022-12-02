from dataclasses import dataclass
from typing import Union, List, Tuple, Callable

from nltk import tokenize
from ..index.index import Index
from ..util import map_with_len
import numpy as np
import pandas as pd
import rank_bm25


@dataclass
class InMemoryBM25Index(Index):

    _index: rank_bm25.BM25
    _metadata: pd.DataFrame
    _tokenize_fn: Callable[[str], List[str]]

    def build(
        corpus: Union[pd.Series, List[str], List[List[str]]],
        metadata: pd.DataFrame,
        tokenize_fn=tokenize.wordpunct_tokenize,
        bm25_cls=rank_bm25.BM25Okapi,
    ):
        if type(corpus) is pd.Series:
            ranker_corpus = map_with_len(tokenize_fn, corpus.to_list())
        elif type(corpus) is list and type(corpus[0]) is str:
            ranker_corpus = map_with_len(tokenize_fn, corpus)
        else:
            # assume corpus is already tokenized
            ranker_corpus = corpus
        _index = bm25_cls(ranker_corpus)
        return InMemoryBM25Index(_index, metadata.reset_index(drop=True), tokenize_fn)

    def find_similar_raw(
        self, query_object: Union[str, List[str]], n_returned: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform nearest neighbor query on index,
        but return only information of retrieved items' index positions and distance

        Parameters
        ----------
        query_object : numpy array (of shape (dimensionality,))
            object for which nearest neighbors are found

        n_returned : int
            number of returned most similar objects


        Returns
        -------
        indices : iterable of int
            indices of nearest neighbors

        distances : iterable of float
            distances between query_object and nearest neighbors

        """
        query_object = (
            self._tokenize_fn(query_object)
            if type(query_object) is str
            else query_object
        )
        corpus_scores = pd.Series(self._index.get_scores(query_object))
        max_score_series = corpus_scores.nlargest(n_returned)
        return max_score_series.index, max_score_series.values

    def dimensionality(self):
        return None

    def metadata(self):
        return self._metadata

    def _get_config(self) -> dict:
        pass

    def _save_index(self, path: str):
        pass

    def _load_from_disk(self, path: str, config: dict, metadata: pd.DataFrame):
        pass
