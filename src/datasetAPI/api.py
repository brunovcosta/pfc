import matplotlib.pyplot as plt


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
