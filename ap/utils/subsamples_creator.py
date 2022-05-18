import random
import json

from pathlib import Path
from itertools import product


class MakeSubsamples:
    """
    Class for saving subsamples of document indices.

    Args:
        languages (list):
            list of str of languages to use in subsamples
        path_to_save_subsamples (str):
            path to file in json format to save subsamples
        path_to_data (str):
            path to data for which subsamples will be created
            expected data in txt format
        path_rubrics (str):
            path to json file with rubrics of documents
        mode ('test' or 'wiki'):
            kind of data for which subsamples will be created
        subsample_size (int):
            The size of the subsample. The default is 1000.
    """
    def __init__(
        self,
        languages: list,
        path_to_save_subsamples: str,
        path_to_data: str,
        path_rubrics: str,
        mode: str,
        subsample_size: int = 1000
    ):
        with open(path_rubrics) as file:
            self._rubrics = json.load(file)

        self._languages = languages
        self._all_ids = list(self._rubrics.keys())
        self._mode = mode
        self._test_ids = dict()
        self._subsample_size = subsample_size
        for lang in self._languages:
            with open(Path(path_to_data).joinpath(
                    f'{self._mode}_{lang}_120k.txt')) as file:
                vw_doc_list = file.readlines()
                self._test_ids[lang] = [vw_doc.split()[0]
                                        for vw_doc in vw_doc_list]
        self._path_to_save_subsamples = path_to_save_subsamples

    def _get_subsample(
        self,
        doc_id: str,
        lang_original: str,
        lang_source: str,
        subsample_size: int
    ) -> list:
        doc_rubric = self._rubrics[doc_id]
        test_sample_ids_source = self._test_ids[lang_source]
        test_sample_ids_original = self._test_ids[lang_original]
        test_sample_ids = list(set(test_sample_ids_source).intersection(
            set(test_sample_ids_original)))
        test_sample_ids = list(set(test_sample_ids).intersection(set(self._all_ids)))
        if len(test_sample_ids) == 0:
            ids_same_rubric = []
            ids_other_rubric = []
            return [ids_same_rubric, ids_other_rubric]

        if subsample_size < len(test_sample_ids):
            subsample_size = len(test_sample_ids)
        test_sample_ids_with_rubric = {current_doc_id: self._rubrics[current_doc_id]
                                       for current_doc_id in test_sample_ids}
        if doc_id not in test_sample_ids_with_rubric:
            ids_same_rubric = []
            ids_other_rubric = []
            return [ids_same_rubric, ids_other_rubric]
        ids_same_rubric = [current_doc_id
                           for current_doc_id, current_rubric
                           in test_sample_ids_with_rubric.items()
                           if current_rubric == doc_rubric]
        if len(ids_same_rubric) < int(0.1 * subsample_size):
            subsample_size = 10 * len(ids_same_rubric)
        if len(ids_same_rubric):
            ids_same_rubric = random.sample(ids_same_rubric, int(0.1 * subsample_size))
        else:
            ids_same_rubric.append(doc_id)

        ids_other_rubric = [current_doc_id
                            for current_doc_id, current_rubric
                            in test_sample_ids_with_rubric.items()
                            if current_rubric != doc_rubric]
        if len(ids_other_rubric) == 0:
            ids_same_rubric = []
            ids_other_rubric = []
            return [ids_same_rubric, ids_other_rubric]

        if len(ids_other_rubric) < int(0.9 * subsample_size):
            if int(len(ids_other_rubric) / 9) < 1:
                ids_same_rubric = random.sample(ids_same_rubric, 1)
                subsample_size = len(ids_other_rubric)
            else:
                ids_same_rubric = random.sample(ids_same_rubric, int(len(ids_other_rubric) / 9))
                subsample_size = 10 * len(ids_same_rubric)
            ids_other_rubric = random.sample(ids_other_rubric, int(0.9 * subsample_size))
        else:
            ids_other_rubric = random.sample(ids_other_rubric, int(0.9 * subsample_size))

        if doc_id not in ids_same_rubric:
            del ids_same_rubric[-1]
            ids_same_rubric.append(doc_id)

        return [ids_same_rubric, ids_other_rubric]

    def get_subsamples(self):
        """
        Function saving subsamples of document indices.
        """
        for lang_original, lang_source in product(self._languages, self._languages):
            print(lang_original, lang_source)
            subsamples = dict()
            for doc_id in self._test_ids[lang_original]:
                if doc_id in self._rubrics:
                    subsamples[doc_id] = self._get_subsample(
                        doc_id, lang_original, lang_source, self._subsample_size)
            path_to_save = Path(self._path_to_save_subsamples).joinpath(lang_original)
            path_to_save.mkdir(parents=True, exist_ok=True)
            with open(path_to_save.joinpath(f'{lang_source}.json'), 'w') as json_file:
                json.dump(subsamples, json_file)
