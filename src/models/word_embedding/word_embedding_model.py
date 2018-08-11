from gensim.models import KeyedVectors
import tensorflow as tf
from ..base_model import BaseModel
from ...utils import TrainValTensorBoard


class WordEmbeddingModelKeras(BaseModel):

    def __init__(self, random_state=1, frac=1,
                 n_features_per_word=50,
                 dict_name=None,
                 min_number_per_label=0):

        self.n_features_per_word = n_features_per_word

        super().__init__(
            random_state,
            frac,
            dict_name,
            min_number_per_label)

        print("loading gensim model...")
        wordEmbedPath = 'dataset/glove/glove_s{}.txt'.format(
            self.n_features_per_word)
        self.wordEmbedModel = KeyedVectors.load_word2vec_format(
            wordEmbedPath,
            unicode_errors="ignore")

        self.padded_length = min(
            int(self.trainObj.avg_text_length * 2),
            self.max_text_len)

    def __repr__(self):
        return f"{super().__repr__()}_{self.n_features_per_word}dimensions"

    def summary(self, save_summary=False):
        self.get_model().summary()
        if save_summary:
            with open(f'./logs/summaries/{self}_summary.txt','w') as fh:
                self.get_model().summary(print_fn=lambda x: fh.write(x + '\n'))

    def fit(self, save_metrics=False, save_checkpoints=False):
        print(f"fitting model {self}...")

        model = self.get_model()

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam')

        callbacks = []

        if save_checkpoints:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    f'./logs/keras_checkpoints/{self}.hdf5',
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min'))

        if save_metrics:
            callbacks.append(TrainValTensorBoard(
                [self.get_X_input(self.trainObj), self.trainObj.target_one_hot],
                log_dir=f'./logs/tf_logs/{self}',
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
        print(f"loading weights to model {self}...")

        model = self.get_model()

        model.load_weights(f'./logs/keras_checkpoints/{checkpoint_file}.hdf5')

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam')
