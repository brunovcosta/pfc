import numpy as np
from datetime import datetime
import sklearn
import matplotlib.pyplot as plt
from ..datasetAPI import RotaDosConcursos


class BaseModel:

    def __init__(self, random_state=1, frac=1,
                 dict_name=None,
                 min_number_per_label=0):
        print(f"loading dataset to {self}...")
        rota_dos_concursos = RotaDosConcursos(
            frac=frac,
            random_state=random_state,
            dict_name=dict_name,
            min_number_per_label=min_number_per_label)
        train, val, test = rota_dos_concursos.split_in_subsets()
        self.data = {
            "train": train,
            "val": val,
            "test": test
        }

        self.max_text_len = train.max_text_length
        self.target_names = train.target_names
        self.n_categories = len(self.target_names)
        self._model = None
        self._X_inputs = {}

    def __repr__(self):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{type(self).__name__}_{now}"

    def get_model(self):
        """
        Returns the model.
        """
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _build_model(self):
        """
        Returns the model for the first use.
        """
        raise NotImplementedError

    def get_X_input(self, dataObj):
        try:
            return self._X_inputs[dataObj]
        except KeyError:
            self._X_inputs[dataObj] = self._build_X_input(dataObj)
            return self._X_inputs[dataObj]

    def _build_X_input(self, dataObj):
        raise NotImplementedError

    def fit(self, save_metrics=False, save_checkpoints=False):
        raise NotImplementedError

    def one_hot_to_label(self, prediction):
        if isinstance(prediction, str):
            return prediction
        else:
            categoryNum = np.argmax(prediction)
            return self.target_names[categoryNum]

    def inspect_mispredictions(self, subset, max_inspect_number):
        dataObj = self.data[subset]
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

    def get_confusion_matrix(self, subset):
        dataObj = self.data[subset]

        X = self.get_X_input(dataObj)
        y_pred = self.get_model().predict(X)
        y_pred = list(map(self.one_hot_to_label, y_pred))

        cnf_matrix = sklearn.metrics.confusion_matrix(
            dataObj.target.tolist(),
            y_pred)

        return cnf_matrix

    def plot_confusion_matrix(self, subset,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = self.get_confusion_matrix(subset)
        classes = self.target_names

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def save_plots(self):
        def plot_save(subset, normalize):
            plt.figure(figsize=(15, 14))
            title = "Confusion matrix"
            if normalize:
                title += " normalized"
            title = f"{title} - {subset}"
            print(f"saving plot {title}...")
            self.plot_confusion_matrix(
                subset,
                title=title,
                normalize=normalize)
            plt.savefig(f'logs/graph_figures/{self} - {title}.png')
        for subset in ['train', 'val', 'test']:
            for normalize in [True, False]:
                plot_save(subset, normalize)
