from datetime import datetime
from gensim.models import KeyedVectors
from ..base_model import BaseModel
from ...utils import TrainValTensorBoard


class WordEmbeddingModelKeras(BaseModel):

    def __init__(self, random_state=1, frac=1,
                 n_features_per_word=50,
                 dict_name=None,
                 min_number_per_label=0):

        print("WordEmbedModelKeras __init__")

        super(WordEmbeddingModelKeras, self).__init__(
            random_state,
            frac,
            dict_name,
            min_number_per_label)

        self.n_features_per_word = n_features_per_word
        wordEmbedPath = 'dataset/glove/glove_s{}.txt'.format(
            self.n_features_per_word)
        self.wordEmbedModel = KeyedVectors.load_word2vec_format(
            wordEmbedPath,
            unicode_errors="ignore")

    def summary(self):
        self.get_model().summary()

    def fit(self, save_metrics=False):
        model = self.get_model()

        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        log_folder_name = f'run-{now}'

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam')

        if save_metrics:
            callbacks = [TrainValTensorBoard(
                [self.get_X_input(self.trainObj), self.trainObj.target_one_hot],
                log_dir=f'./logs/tf_logs/{log_folder_name}',
                write_graph=True)]
        else:
            callbacks = None

        model.fit(
            self.get_X_input(self.trainObj),
            self.trainObj.target_one_hot,
            validation_data=(
                self.get_X_input(self.testObj),
                self.testObj.target_one_hot),
            epochs=9,
            batch_size=32,
            shuffle=True,
            callbacks=callbacks)
