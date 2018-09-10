from .api import DatasetDistributer
import pandas as pd
import json
import re
import glob
import os
import nltk


class RotaDosConcursos(DatasetDistributer):

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

        super().__init__(
            self.df,
            random_state,
            frac,
            min_number_per_label,
            dict_name)

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
