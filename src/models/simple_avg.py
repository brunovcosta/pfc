import numpy as np
import keras
from gensim.models import KeyedVectors
from ..datasetAPI import RotaDosConcursos


class SimpleAvg:

    def __init__(self):
        self.trainObj = RotaDosConcursos(subset='train')
        self.testObj = RotaDosConcursos(subset='test')

        self.nCategories = len(self.trainObj.target_names)

        self.nFeaturesPerWord = 50
        wordEmbedPath = 'dataset/glove/glove_s{}.txt'.format(
            self.nFeaturesPerWord)
        wordEmbedModel = KeyedVectors.load_word2vec_format(
            wordEmbedPath,
            unicode_errors="ignore")

        self.X_train_avg = self.vector_sentence_to_avg(wordEmbedModel)

        self.X_train_avg = np.array(self.X_train_avg)

    def row_sentence_to_avg(self, row, wordEmbedModel, answer_list):
        """
        Converts a sentence (string) into a list of words (strings). Extracts
        the word2Vec representation of each word and averages its value into
        a single vector encoding the meaning of the sentence.
        """

        words = row.clean_text.lower().split()

        nFeaturesPerWord = len(wordEmbedModel.word_vec('casa'))

        avg = np.zeros((nFeaturesPerWord,))
        total = len(words)
        for w in words:
            try:
                avg += wordEmbedModel.word_vec(w)
            except KeyError:
                total -= 1
        if total != 0:
            avg = avg / total
        else:
            print("Clean text with no words in the embedding model for index {} .".format(row.name))
        answer_list.append(avg)

    def vector_sentence_to_avg(self, wordEmbedModel):
        X_train_avg = []
        self.trainObj.df.apply(self.row_sentence_to_avg, axis=1, args=[wordEmbedModel, X_train_avg])
        return X_train_avg

    def simple_model(self):
        X_input = keras.layers.Input(shape=(self.nFeaturesPerWord,))

        X = keras.layers.Dense(self.nCategories, name='fc')(X_input)
        X = keras.layers.Activation('softmax')(X)
        model = keras.models.Model(
            inputs=X_input,
            outputs=X,
            name='simple_model')

        return model

    def num_to_label(self, categoryNum):
        return self.target_names[categoryNum]

    def execute_model(self):
        model = self.simple_model()
        model.summary()

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        model.fit(
            self.X_train_avg,
            self.trainObj.target_one_hot,
            epochs=50,
            batch_size=32,
            shuffle=True)

        loss, acc = model.evaluate(self.X_train_avg, self.trainObj.target_one_hot)
        print("\nTrain accuracy = ", acc)

        self.target_names = self.trainObj.target_names

        pred = model.predict(self.X_train_avg)
        for i in range(len(self.trainObj.target)):
            categoryNum = np.argmax(pred[i])
            if self.num_to_label(categoryNum) != self.trainObj.target.iloc[i]:
                print("\n\n Text:\n", self.trainObj.text.iloc[i])
                print('\nExpected category:' + self.trainObj.target.iloc[i] + ' prediction: ' + self.num_to_label(categoryNum))