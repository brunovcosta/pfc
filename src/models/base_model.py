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

    def build_model(self):
        """
        Returns the model.
        """
        raise NotImplementedError

    def execute_model(self):
        raise NotImplementedError

    def one_hot_to_label(self, prediction):
        if isinstance(prediction, str):
            return prediction
        else:
            categoryNum = np.argmax(prediction)
            return self.target_names[categoryNum]

    def inspect_mispredictions(self, model, dataObj, X, max_inspect_number):
        pred = model.predict(X)
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
