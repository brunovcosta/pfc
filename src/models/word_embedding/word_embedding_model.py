from datetime import datetime
from gensim.models import KeyedVectors
import tensorflow as tf
from ..base_model import BaseModel
from ...utils import TrainValTensorBoard


class WordEmbeddingModelKeras(BaseModel):

    def __init__(self, random_state=1, frac=1,
                 n_features_per_word=50,
                 dict_name=None,
                 min_number_per_label=0):

        super(WordEmbeddingModelKeras, self).__init__(
            random_state,
            frac,
            dict_name,
            min_number_per_label)

        print("loading gensim model...")
        self.n_features_per_word = n_features_per_word
        wordEmbedPath = 'dataset/glove/glove_s{}.txt'.format(
            self.n_features_per_word)
        self.wordEmbedModel = KeyedVectors.load_word2vec_format(
            wordEmbedPath,
            unicode_errors="ignore")

    def summary(self):
        self.get_model().summary()

    def fit(self, save_metrics=False, save_checkpoints=False):
        model = self.get_model()

        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        log_name = f'run-{now}'

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam')

        callbacks = []

        if save_checkpoints:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    f'./logs/keras_checkpoints/{log_name}.hdf5',
                    monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='max'))

        if save_metrics:
            callbacks.append(TrainValTensorBoard(
                [self.get_X_input(self.trainObj), self.trainObj.target_one_hot],
                log_dir=f'./logs/tf_logs/{log_name}',
                write_graph=True))

        if not callbacks:
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

    def load_model(self, checkpoint_file):
        model = self.get_model()

        model.load_weights(f'./logs/keras_checkpoints/{checkpoint_file}.hdf5')

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam')
