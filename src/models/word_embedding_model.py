from datetime import datetime
from gensim.models import KeyedVectors
from .base_model import BaseModel
from ..utils import TrainValTensorBoard


class WordEmbeddingModelKeras(BaseModel):

    def __init__(self, random_state=1, frac=1,
                 n_features_per_word=50,
                 group_labels=False,
                 min_number_per_label=0):

        super(WordEmbeddingModelKeras, self).__init__(
            random_state,
            frac,
            group_labels,
            min_number_per_label)

        self.n_features_per_word = n_features_per_word
        wordEmbedPath = 'dataset/glove/glove_s{}.txt'.format(
            self.n_features_per_word)
        self.wordEmbedModel = KeyedVectors.load_word2vec_format(
            wordEmbedPath,
            unicode_errors="ignore")

        self.X_train = self.generate_X_input(self.trainObj)
        self.X_test = self.generate_X_input(self.testObj)

    def execute_model(self):
        model = self.build_model()
        model.summary()

        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        log_folder_name = f'run-{now}'

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam')

        tbCallBack = TrainValTensorBoard(
            [self.X_train, self.trainObj.target_one_hot],
            log_dir=f'./tf_logs/{log_folder_name}',
            write_graph=True)

        model.fit(
            self.X_train,
            self.trainObj.target_one_hot,
            validation_data=(self.X_test, self.testObj.target_one_hot),
            epochs=9,
            batch_size=32,
            shuffle=True,
            callbacks=[tbCallBack])

        #self.inspect_mispredictions(model, self.trainObj, self.X_train, 40)
        self.inspect_mispredictions(model, self.testObj, self.X_test, 40)

    def generate_X_input(self, dataObj):
        raise NotImplementedError
