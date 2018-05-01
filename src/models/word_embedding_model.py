from .base_model import BaseModel
from gensim.models import KeyedVectors

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

    def execute_model(self):
        model = self.build_model()
        model.summary()

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        model.fit(
            self.X_train_avg,
            self.trainObj.target_one_hot,
            epochs=9,
            batch_size=32,
            shuffle=True)

        loss, acc = model.evaluate(self.X_train_avg, self.trainObj.target_one_hot)
        print("\nTrain accuracy = ", acc)

        loss, acc = model.evaluate(self.X_test_avg, self.testObj.target_one_hot)
        print("\nTest accuracy = ", acc)

        #self.inspect_mispredictions(model, self.trainObj, self.X_train_avg, 40)
        self.inspect_mispredictions(model, self.testObj, self.X_test_avg, 40)
