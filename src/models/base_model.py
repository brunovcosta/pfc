import numpy as np
from ..datasetAPI import RotaDosConcursos


class BaseModel:

    def __init__(self, random_state=1, frac=1,
                 group_labels=False,
                 min_number_per_label=0):
        self.trainObj = RotaDosConcursos(
            subset='train',
            frac=frac,
            random_state=random_state,
            group_labels=group_labels,
            min_number_per_label=min_number_per_label)
        self.testObj = RotaDosConcursos(
            subset='test',
            frac=frac,
            random_state=random_state,
            group_labels=group_labels,
            min_number_per_label=min_number_per_label)

        self.max_len = self.trainObj.max_text_length("text")
        self.target_names = self.trainObj.target_names
        self.n_categories = len(self.target_names)

    def build_model(self):
        """
        Returns the model.
        """
        pass

    def execute_model(self):
        pass

    def num_to_label(self, categoryNum):
        return self.target_names[categoryNum]

    def inspect_mispredictions(self, model, dataObj, X, max_inspect_number):
        pred = model.predict(X)
        mispredictions_count = 0
        for i in range(len(dataObj.target)):
            categoryNum = np.argmax(pred[i])
            if self.num_to_label(categoryNum) != dataObj.target.iloc[i]:
                print("\n\n Text:\n", dataObj.text.iloc[i])
                print('\nExpected category: {}\nPrediction: {}'.format(
                    dataObj.target.iloc[i],
                    self.num_to_label(categoryNum)))
                mispredictions_count += 1
                if mispredictions_count > max_inspect_number:
                    break
