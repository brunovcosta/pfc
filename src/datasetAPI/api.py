from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


class Dataset:

    def __init__(self, df, one_hot, target_names, max_text_len, avg_text_len, median_text_len):
        """
        df: pandas DataFrame with the following columns:
            text, label, splitted_text

        one_hot: pandas DataFrame
            One hot encoding for each data sample.

        target_names: list of strings
            List with exactly one string for each possible label.

        max_text_len: integer

        avg_text_len: float

        median_text_len: integer
        """
        self.df = df
        self._one_hot = one_hot
        self._target_names = target_names
        self._max_text_len = max_text_len
        self._avg_text_len = avg_text_len
        self._median_text_len = median_text_len

    @property
    def target_names(self):
        return self._target_names

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

    @property
    def avg_text_length(self):
        return self._avg_text_len

    @property
    def median_text_length(self):
        return self._median_text_len

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


class DatasetDistributer(Dataset):

    def __init__(self, df, random_state=1,
                 frac=1, min_number_per_label=0,
                 dict_name=None):
        """
        df: pandas DataFrame with the following columns:
            text, label, splitted_text

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
        self.df = df
        self._random_state = random_state

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
