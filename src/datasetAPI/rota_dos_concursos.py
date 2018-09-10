import json
import re
import glob
import os
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .api import Dataset


class RotaDosConcursos(Dataset):

    def __init__(self, random_state=1,
                 frac=1, min_number_per_label=0,
                 dict_name=None):
        """
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
        self._random_state = random_state

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
            self.df.to_csv(csv_path)

        if dict_name is not None:
            self.df.apply(
                self._apply_group_labels,
                axis=1,
                args=[self._generate_group_labels_dict(dict_name)])
        self._drop_labels_with_insufficient_data(min_number_per_label)
        self.df = self.df.sample(frac=frac, random_state=self._random_state)
        self._one_hot = pd.get_dummies(self.df['label'])
        self._save_text_properties()

    def split_in_subsets(self):
        """
        Returns a tuple with three Dataset objects, one for each
        of the following sets: training, validation, and test.
        """

        df_train, df_val_test = train_test_split(
            self.df,
            test_size=0.2,
            random_state=self._random_state)
        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=self._random_state)

        one_hot_train, one_hot_val_test = train_test_split(
            self._one_hot,
            test_size=0.2,
            random_state=self._random_state)
        one_hot_val, one_hot_test = train_test_split(
            one_hot_val_test,
            test_size=0.5,
            random_state=self._random_state)

        dataset_parameters = {
            "target_names": self.target_names,
            "max_text_len": self.max_text_length,
            "avg_text_len": self.avg_text_length,
            "median_text_len": self.median_text_length,
        }
        trainObj = Dataset(df_train, one_hot_train, **dataset_parameters)
        valObj = Dataset(df_val, one_hot_val, **dataset_parameters)
        testObj = Dataset(df_test, one_hot_test, **dataset_parameters)

        return trainObj, valObj, testObj

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

    def _save_text_properties(self):
        splitted_text_len = list(map(len, self.df.splitted_text))
        splitted_text_len = np.array(splitted_text_len)
        self._max_text_len = max(splitted_text_len)
        self._avg_text_len = np.mean(splitted_text_len)
        self._median_text_len = np.median(splitted_text_len)
        self._target_names = self.target_one_hot.axes[1]
