import numpy as np
import keras
from gensim.models import KeyedVectors
from ..datasetAPI import RotaDosConcursos


class SimpleAvg:
    def max_word_length(X):
        splittedXlen = map(lambda x: len(x.split()), X)
        return max(splittedXlen)

    def sentence_to_avg(sentence, wordEmbedModel):
        """
        Converts a sentence (string) into a list of words (strings). Extracts
        the word2Vec representation of each word
        and averages its value into a single vector encoding the meaning of the
        sentence.

        Returns:
        avg -- average vector encoding information about the sentence,
        numpy-array
        """

        words = sentence.lower().split()

        nFeaturesPerWord = len(wordEmbedModel.word_vec('casa'))

        avg = np.zeros((nFeaturesPerWord,))

        for w in words:
            avg += wordEmbedModel.word_vec(w)  # !!!!!!!!!!!Adicionar <UNK> ao vocabulario (media de todas as palavras?)
        avg = avg / len(words)

        return avg

    def vector_sentence_to_avg(self, wordEmbedModel):
        X = []

        for sentence in self.trainObj.text:
            X.append(self.sentence_to_avg(sentence, wordEmbedModel))

        return X

    def simple_model(input_shape, nCategories):
        X_input = keras.layers.Input(input_shape)

        X = keras.layers.Dense(nCategories, name='fc')(X_input)
        X = keras.layers.Activation('softmax')(X)
        model = keras.models.Model(
            inputs=X_input,
            outputs=X,
            name='simple_model')

        return model

    def label_to_category(target_names, categoryNum):
        return target_names[categoryNum]

    def __init__(self):
        self.trainObj = RotaDosConcursos(subset='train')
        self.testObj = RotaDosConcursos(subset='test')

        nCategories = len(self.trainObj.target_names)

        self.nFeaturesPerWord = 50
        wordEmbedPath = '../../../dataset/glove/glove_s{}.txt'.format(
            str(self.nFeaturesPerWord))
        wordEmbedModel = KeyedVectors.load_word2vec_format(
            wordEmbedPath,
            unicode_errors="ignore")

        wordEmbedModel.word_vec('casa')

        X_train_avg = self.vector_sentence_to_avg(
            self,
            wordEmbedModel)

        model = self.simple_model(X_train_avg.shape, nCategories)
        model.summary()

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        model.fit(
            X_train_avg,
            self.trainObj.target_one_hot,
            epochs=50,
            batch_size=32,
            shuffle=True)

        loss, acc = model.evaluate(X_train_avg, self.trainObj.target_one_hot)
        print("\nTrain accuracy = ", acc)

        self.target_names = self.trainObj.target_names

        pred = model.predict(X_train_avg)
        for i in range(len(X_train_avg)):
            categoryNum = np.argmax(pred[i])
            if categoryNum != np.argmax(Y_oh_train[i]):
                print("\n\n Text:\n", X_train[i])
                print('\nExpected category:' + self.trainObj.target.iloc[i] + ' prediction: ' + self.label_to_category(categoryNum).strip())
