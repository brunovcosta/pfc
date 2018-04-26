import json
import re
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class RotaDosConcursos:

    def __init__(self, random_state=1, subset='all', frac=1):
        """
        subset : 'train' or 'test', 'all', optional
            Select the dataset to load: 'train' for the training set, 'test'
            for the test set, 'all' for both, with shuffled ordering.

        random_state : numpy random number generator or seed integer
            Used to shuffle the dataset.

        frac: float
            0 < frac <=1
            Fraction of the data that is going to be used.
        """

        csv_path = 'dataset/rota_dos_concursos.csv'

        if os.path.isfile(csv_path):
            self.df = pd.read_csv(csv_path,
                                  encoding='UTF-8',
                                  index_col=0,
                                  dtype={"text": str,
                                         "label": str})
        else:
            texts = []
            clean_texts = []
            labels = []
            ids = []

            dataset_path = 'dataset/rawData/**/*.json'
            for filename in glob.iglob(dataset_path, recursive=True):
                filename = filename.replace("\\", '/')

                try:
                    data = json.load(open(filename), encoding='UTF-8')

                    # empty text or subject
                    no_subject_path = (len(data["subject_path"]) == 0)
                    no_valid_text = (len(data["text"]) < 20)
                    if no_subject_path or no_valid_text:
                        continue

                    # label
                    labels.append(data["subject_path"][0].strip())

                    # text
                    text = ' '.join(data["text"].splitlines())
                    texts.append(text.strip())

                    # clean_text
                    word_regex = r'(\w+(-\w+)?)'
                    clean_tokens = re.findall(word_regex, text, re.UNICODE)
                    clean_text = ' '.join([pair[0] for pair in clean_tokens])
                    clean_texts.append(clean_text)

                    # id
                    ids.append(data["id"])
                except UnicodeDecodeError:
                    # Windows error: can't decode byte 0x81
                    print("byte 0x81 error {}".format(filename))
                except json.JSONDecodeError:
                    # empty json file!!!
                    print("JSON error {}".format(filename))

            self.df = pd.DataFrame({
                "text": texts,
                "clean_text": clean_texts,
                "label": labels
            }, index=ids)

            indexes_to_drop = self.df.loc[self.df.clean_text == ""].index
            self.df.drop(indexes_to_drop, inplace=True)
            self.df.drop_duplicates(inplace=True)
            self.df.reset_index(inplace=True, drop=True)        # Temporary solution (crawler change TODO)
            self.df.to_csv(csv_path)

        self.df.drop_duplicates(inplace=True)
        self.df = self.df.sample(frac=frac, random_state=random_state)
        self._one_hot = pd.get_dummies(self.df['label'])

        def max_text_length(text_column):
            """
            text_column : 'text' or 'clean_text'
            """
            splitted_text_len = map(lambda text: len(text.split()), self.df.loc[:, text_column])
            return max(splitted_text_len)

        self.max_text_length_dict = {
            'text': max_text_length('text'),
            'clean_text': max_text_length('clean_text')
        }

        if subset == 'train':
            self.df, _ = train_test_split(
                self.df,
                test_size=0.2,
                random_state=random_state)
            self._one_hot, _ = train_test_split(
                self._one_hot,
                test_size=0.2,
                random_state=random_state)

        if subset == 'test':
            _, self.df = train_test_split(
                self.df,
                test_size=0.2,
                random_state=random_state)

            _, self._one_hot = train_test_split(
                self._one_hot,
                test_size=0.2,
                random_state=random_state)

    def max_text_length(self, text_column):
        return self.max_text_length_dict[text_column]

    @property
    def target_names(self):
        return self.target_one_hot.axes[1]

    @property
    def target(self):
        return self.df['label']

    @property
    def text(self):
        return self.df['text']

    @property
    def clean_text(self):
        return self.df['clean_text']

    @property
    def target_one_hot(self):
        return self._one_hot
