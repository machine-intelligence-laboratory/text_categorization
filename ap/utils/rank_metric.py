import os
from pathlib import Path

import artm
import numpy as np


class RankingByModel:
    def __init__(self, model_path, metric):
        """
        Class for ranking document search between language pairs.

        Parameters
        ----------

        model_path: str
            a path to the artm model directory
        metric: callable
            a way to measure norm of a matrix of vectors
            for example init as:
            metric = np.linalg.norm
        """
        self._model = artm.load_artm_model(model_path)
        self._metric = metric

    def _get_document_embeddings(self, path_to_data):
        if os.path.isfile(path_to_data):
            path_to_batches = os.path.join(
                os.path.dirname(path_to_data), Path(path_to_data).stem + "_rank_batches"
            )
            bv = artm.BatchVectorizer(
                data_path=path_to_data,
                data_format="vowpal_wabbit",
                target_folder=path_to_batches,
            )

        elif len(os.listdir(path_to_data)) > 0:
            bv = artm.BatchVectorizer(data_path=path_to_data, data_format="batches",)
        else:
            raise ValueError("Unknown data format")

        theta = self._model.transform(batch_vectorizer=bv)
        return theta.columns, theta

    def _rank(self, search_indices, vectors_first, vectors_secnd, kwargs=None):
        average_position = []
        for search_num in range(len(search_indices)):
            difference = vectors_secnd - vectors_first[search_num]
            vectors_norm = self._metric(difference, **kwargs)
            rating = search_indices[np.argsort(vectors_norm)].copy()
            average_position.append(
                np.argwhere(rating == search_indices[search_num])[0][0] + 1
            )
        return np.mean(average_position)

    def get_ranking(self, data_lang_one, data_lang_two, kwargs=None):
        """
        Function returning average position of search documents in the other language.

        Parameters
        ----------
        data_lang_one: str
            path to folder with batches or path to vw file
        data_lang_two: str
            path to folder with batches or path to vw file
        """
        idx_one, theta_one = self._get_document_embeddings(data_lang_one)
        idx_two, theta_two = self._get_document_embeddings(data_lang_two)

        assert len(idx_one) == len(set(idx_one))
        assert len(idx_two) == len(set(idx_two))

        search_indices = idx_one.intersection(idx_two)

        vectors_one = theta_one[search_indices].values.T
        vectors_two = theta_two[search_indices].values.T
        del theta_one, theta_two

        ranking_first = self._rank(search_indices, vectors_one, vectors_two, kwargs)
        ranking_secnd = self._rank(search_indices, vectors_two, vectors_one, kwargs)

        return ranking_first, ranking_secnd, len(search_indices)
