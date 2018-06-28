import numpy as np
from ..datasetAPI import RotaDosConcursos


class BaseModel:

    def __init__(self, random_state=1, frac=1,
                 dict_name=None,
                 min_number_per_label=0):
        self.trainObj = RotaDosConcursos(
            subset='train',
            frac=frac,
            random_state=random_state,
            dict_name=dict_name,
            min_number_per_label=min_number_per_label)
        self.testObj = RotaDosConcursos(
            subset='test',
            frac=frac,
            random_state=random_state,
            dict_name=dict_name,
            min_number_per_label=min_number_per_label)

        self.max_len = self.trainObj.max_text_length("text")
        self.target_names = self.trainObj.target_names
        self.n_categories = len(self.target_names)
        self.model = None
        self.X_inputs = {}

    def get_model(self):
        """
        Returns the model.
        """
        if self.model is None:
            self.model = self._build_model()
        return self.model

    def _build_model(self):
        """
        Returns the model for the first use.
        """
        raise NotImplementedError

    def get_X_input(self, dataObj):
        try:
            return self.X_inputs[dataObj]
        except KeyError:
            self.X_inputs[dataObj] = self._build_X_input(dataObj)
            return self.X_inputs[dataObj]

    def _build_X_input(self, dataObj):
        raise NotImplementedError

    def fit(self, save_metrics=False):
        raise NotImplementedError

    def one_hot_to_label(self, prediction):
        if isinstance(prediction, str):
            return prediction
        else:
            categoryNum = np.argmax(prediction)
            return self.target_names[categoryNum]

    def inspect_mispredictions(self, dataObj_name, max_inspect_number):
        if dataObj_name == 'train':
            dataObj = self.trainObj
        elif dataObj_name == 'test':
            dataObj = self.testObj
        X = self.get_X_input(dataObj)
        pred = self.get_model().predict(X)
        mispredictions_count = 0
        for index, actual_target in enumerate(dataObj.target):
            if self.one_hot_to_label(pred[index]) != actual_target:
                print("\n\n Text:\n", dataObj.text.iloc[index])
                print('\nExpected category: {}\nPrediction: {}'.format(
                    actual_target,
                    self.one_hot_to_label(pred[index])))
                mispredictions_count += 1
                if mispredictions_count > max_inspect_number:
                    break
