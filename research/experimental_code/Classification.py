import json
import os
import artm
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib
import numpy
import pandas
import typing


class GetAccuracy:

    def __init__(self, config_classification_quality):
        """
        Initialize variables.

        Parameters
        ----------
        config_classification_quality: json
            json file with the following fields:
                path_model: str
                    A path to folder with an artm model.
                path_test: str
                    A path to test data in txt format.
                rubrics_train: str
                    A path to the file with rubric for train data in json format.
                rubrics_test: str
                    The path to the file with rubric for test data in json format.
                path_train: str
                    A path to folder with train data.
                path_to_save_train_theta: list of str
                    A path to folder to save theta for model.
                classifier: sklearn.linear_model
                    sklearn model for classification
        """
        path_model = config_classification_quality['path_model']
        self.model = artm.load_artm_model(path_model)
        self.model_name = str(Path(path_model).name)
        self.path_test = config_classification_quality['path_test']
        self.path_train = config_classification_quality['path_train']
        self.recalculate_train_theta = config_classification_quality['recalculate_train_theta']
        self.path_to_save_train_theta = Path(
            config_classification_quality['path_to_save_train_theta']
        ).joinpath(f'theta_{self.model_name}.joblib')
        self.classifier = config_classification_quality['classifier']

        with open(config_classification_quality['path_rubrics_train']) as f:
            rubrics = json.load(f)
        self.rubrics_train = rubrics

        with open(config_classification_quality['path_rubrics_test']) as f:
            rubrics = json.load(f)
        self.rubrics_test = rubrics

    def get_accuracy(self) -> typing.Tuple[pandas.core.frame.DataFrame,
                                           pandas.core.frame.DataFrame,
                                           numpy.ndarray]:
        """
        Calculate accuracy, using given classifier.

        Parameters
        ----------

        Return
        ----------
        Pandas DataFrame with accuracy for each model.
        """
        if self.recalculate_train_theta:
            if Path(self.path_to_save_train_theta).exists():
                Path(self.path_to_save_train_theta).unlink()
            theta_train = self._get_theta(self.path_train)
            joblib.dump(theta_train, self.path_to_save_train_theta)
        else:
            theta_train = joblib.load(self.path_to_save_train_theta)
        theta_test = self._get_theta(self.path_test)
        accuracy, y_predictions = self._calculate_accuracy(theta_train, theta_test)

        return accuracy, theta_test, y_predictions

    def _get_theta(self, path_to_data: str) -> typing.Tuple[pandas.core.frame.DataFrame,
                                                            pandas.core.frame.DataFrame]:
        if os.path.isfile(path_to_data):
            path_to_batches = os.path.join(
                os.path.dirname(path_to_data),
                Path(path_to_data).stem + '_rank_batches')
            if Path(path_to_batches).exists():
                batches_list = list(Path(path_to_batches).iterdir())
                if batches_list:
                    for batch in batches_list:
                        batch.unlink()
            train_batch_vectorizer = artm.BatchVectorizer(data_path=path_to_data,
                                                          data_format='vowpal_wabbit',
                                                          batch_size=10000,
                                                          target_folder=path_to_batches)

        elif len(os.listdir(path_to_data)) > 0:
            train_batch_vectorizer = artm.BatchVectorizer(
                data_path=path_to_data,
                data_format='batches'
            )
        else:
            raise ValueError('Unknown data format')

        theta = self.model.transform(batch_vectorizer=train_batch_vectorizer).T
        theta = theta.loc[set(self.rubrics_train.keys()).intersection(theta.index)]

        return theta

    def _calculate_accuracy(
        self,
        X_train: pandas.core.frame.DataFrame,
        X_test: pandas.core.frame.DataFrame
    ) -> typing.Tuple[numpy.float64, numpy.ndarray]:

        labels_train = [int(self.rubrics_train[title]) for title in X_train.index]
        labels_test = [int(self.rubrics_test[title]) for title in X_test.index]
        self.classifier.fit(X_train, labels_train)
        y_predictions = self.classifier.predict(X_test)
        accuracy = accuracy_score(labels_test, y_predictions)

        return accuracy, y_predictions
