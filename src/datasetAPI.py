import json
import re
import glob
import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class RotaDosConcursos:

    def __init__(self, random_state=1,
                 subset='all', frac=1,
                 min_number_per_label=0,
                 dict_name=None):
        """
        subset: 'train' or 'test', 'all', optional
            Select the dataset to load: 'train' for the training set, 'test'
            for the test set, 'all' for both, with shuffled ordering.

        random_state: numpy random number generator or seed integer
            Used to shuffle the dataset.

        frac: float
            0 < frac <=1
            Fraction of the data that is going to be used.

        min_number_per_label: integer
            Minimum number of samples for each category, samples from categories
            with less the minimum are dropped.

        dict_name: string
            If None, labels with similar subject are not grouped. Otherwise, it
            is used as the name of the JSON file with the mapping of the labels
            to be grouped.
            ("default.json" is a recommended dictionary)
        """

        csv_path = 'dataset/rota_dos_concursos.csv'

        if os.path.isfile(csv_path):
            def splitted_text_converter(splitted_text):
                splitted_text = splitted_text.strip("[]").split(", ")
                splitted_text = list(map(
                    lambda x: x.strip("\'"),
                    splitted_text))
                return splitted_text

            self.df = pd.read_csv(
                csv_path,
                encoding='UTF-8',
                index_col=0,
                converters={
                    "text": str,
                    "label": str,
                    "splitted_text": splitted_text_converter})
        else:
            texts = []
            splitted_texts = []
            labels = []
            ids = []

            dataset_path = 'dataset/rawData/**/*.json'
            for filename in glob.iglob(dataset_path, recursive=True):
                filename = filename.replace("\\", '/')
                try:
                    self._json_extraction(filename, texts, splitted_texts, labels, ids)
                except UnicodeDecodeError:
                    # Windows error: can't decode byte 0x81
                    print("byte 0x81 error {}".format(filename))
                except json.JSONDecodeError:
                    # empty json file!!!
                    print("JSON error {}".format(filename))

            self.df = pd.DataFrame({
                "text": texts,
                "splitted_text": splitted_texts,
                "label": labels
            }, index=ids)

            self._drop_inconsistencies()
            self.df.reset_index(inplace=True, drop=True)        # Temporary solution (crawler change TODO)
            self.df.to_csv(csv_path)

        self._drop_inconsistencies()
        if dict_name is not None:
            self.df.apply(
                self._apply_group_labels,
                axis=1,
                args=[self._generate_group_labels_dict(dict_name)])
        self._drop_labels_with_insufficient_data(min_number_per_label)
        self.df = self.df.sample(frac=frac, random_state=random_state)
        self._one_hot = pd.get_dummies(self.df['label'])
        self._save_max_text_length()
        self.df, self._one_hot = self._save_subset(subset, random_state)

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
    def splitted_text(self):
        return self.df['splitted_text']

    @property
    def target_one_hot(self):
        return self._one_hot

    @property
    def max_text_length(self):
        return self._max_text_len

    def save_pie_graph(self):
        if len(self.target_names) <= 10:
            autopct = '%1.1f%%'
        else:
            autopct = None
        plt.figure(figsize=(16, 14))
        self.target.value_counts().plot(
            kind='pie',
            autopct=autopct,
            legend=False,
            title='labels distribution',
            fontsize=14)
        plt.savefig(f'logs/graph_figures/pie_graph.png')

    def _json_extraction(self, filename, texts, splitted_texts, labels, ids):
        data = json.load(open(filename), encoding='UTF-8')

        # empty text or subject
        no_subject_path = (len(data["subject_path"]) == 0)
        no_valid_text = (len(data["text"]) < 20)
        image_content = int(data["image_count"]) > 0
        if no_subject_path or no_valid_text or image_content:
            return

        # label
        labels.append(data["subject_path"][0].strip())

        # text
        text = data["text"]
        for alternative in data["alternatives"]:
            text += " " + alternative
        text = ' '.join(text.splitlines())
        # cleaning text
        word_regex = r'(\w+(-\w+)?)'
        clean_tokens = re.findall(word_regex, text, re.UNICODE)
        text = ' '.join([pair[0] for pair in clean_tokens])
        text = text.lower()
        texts.append(text)

        # splitted_text
        splitted_text = nltk.tokenize.word_tokenize(text)
        splitted_texts.append(splitted_text)

        # id
        ids.append(data["id"])

    def _drop_inconsistencies(self):
        # Empty text after cleaning
        indexes_to_drop = self.df[self.df.text == ""].index
        self.df.drop(indexes_to_drop, inplace=True)

        #Duplicates
        self.df.drop_duplicates(
            subset=["text", "label"],
            inplace=True)

        #Mixed categories
        boolean_drop = (self.target == "Conhecimentos Específicos de um determinado Cargo/Área")
        indexes_to_drop = self.target[boolean_drop].index
        self.df.drop(
            labels=indexes_to_drop,
            axis=0,
            inplace=True)

    def _apply_group_labels(self, row, convert_dict):
        row.label = convert_dict[row.label]
        return row

    def _generate_group_labels_dict(self, dict_name):
        target_names = self.target.value_counts().index
        convert_dict = {
            label: label for label in target_names}

        with open(f"src/group_dictionaries/{dict_name}", encoding="UTF-8") as file:
            external_dict = json.load(file)
        for key, group_list in external_dict.items():
            for label in group_list:
                convert_dict[label] = key

        return convert_dict

    def _drop_labels_with_insufficient_data(self, min_number_per_label):
        labels_to_remove = []
        label_count = self.target.value_counts()
        for pos, label_num in enumerate(label_count):
            if label_num < min_number_per_label:
                labels_to_remove.append(label_count.index[pos])
        boolean_drop = self.target.isin(labels_to_remove)
        indexes_to_drop = self.target[boolean_drop].index
        self.df.drop(indexes_to_drop, inplace=True)

    def _save_max_text_length(self):
        splitted_text_len = list(map(len, self.df.splitted_text))
        self._max_text_len = max(splitted_text_len)

    def _save_subset(self, subset, random_state):
        if subset == 'all':
            return self.df, self._one_hot

        df_train, df_test = train_test_split(
            self.df,
            test_size=0.2,
            random_state=random_state)
        _one_hot_train, _one_hot_test = train_test_split(
            self._one_hot,
            test_size=0.2,
            random_state=random_state)

        if subset == 'train':
            return df_train, _one_hot_train

        if subset == 'test':
            return df_test, _one_hot_test
