import json
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class RotaDosConcursos:

    def __init__(self, random_state=1, subset='all'):
        """
        subset : 'train' or 'test', 'all', optional
            Select the dataset to load: 'train' for the training set, 'test'
            for the test set, 'all' for both, with shuffled ordering.

        random_state : numpy random number generator or seed integer
            Used to shuffle the dataset.
        """

        my_path = os.path.abspath(os.path.dirname(__file__))
        csv_path = os.path.join(
                my_path,
                '../../dataset/rota_dos_concursos.csv')

        if os.path.isfile(csv_path):
            self.df = pd.read_csv(
                    csv_path,
                    encoding='UTF-8',
                    index_col=0)
        else:
            texts = []
            labels = []
            ids = []

            for filename in glob.iglob('../../dataset/rawData/**/*.json', recursive=True):
                filename = filename.replace("\\", '/')

                try:
                    data = json.load(open(filename), encoding='UTF-8')
                    if len(data["subject_path"]) == 0 or len(data["text"]) == 0:
                        continue
                    labels.append(data["subject_path"][0].strip())
                    text = ' '.join(data["text"].splitlines())
                    texts.append(text.strip())
                    ids.append(data["id"])
                except UnicodeDecodeError:
                    # Windows error: can't decode byte 0x81
                    print("byte 0x81 error {}".format(filename))
                except json.JSONDecodeError:
                    # empty json file!!!
                    print("JSON error {}".format(filename))

            self.df = pd.DataFrame({
                "text": texts,
                "label": labels
            }, index=ids)

            self.df.to_csv(csv_path)

        self._one_hot = pd.get_dummies(self.df['label'])

        if subset == 'train':
            self.df, _ = train_test_split(self.df, test_size=0.2, random_state=random_state)
            self._one_hot, _ = train_test_split(self._one_hot, test_size=0.2, random_state=random_state)

        if subset == 'test':
            _, self.df = train_test_split(self.df, test_size=0.2, random_state=random_state)
            _, self._one_hot = train_test_split(self._one_hot, test_size=0.2, random_state=random_state)


    @property
    def target_names(self):
        target_names = self.df['label'].unique()
        target_names.sort()
        return target_names


    @property
    def target(self):
        return self.df['label']


    @property
    def text(self):
        return self.df['text']


    @property
    def target_one_hot(self):
        return self._one_hot


if __name__ == '__main__':
    obj = RotaDosConcursos(subset='train')

    print(obj.target)
