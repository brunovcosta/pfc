import numpy as np
import keras
from gensim.models import KeyedVectors
from ..datasetAPI import RotaDosConcursos


class SimpleAvg:

    def __init__(self, n_features_per_word=50, random_state=1,
                 group_labels=False, min_number_per_label=0):
        self.trainObj = RotaDosConcursos(
            subset='train',
            random_state=random_state,
            group_labels=group_labels,
            min_number_per_label=min_number_per_label)
        self.testObj = RotaDosConcursos(
            subset='test',
            random_state=random_state,
            group_labels=group_labels,
            min_number_per_label=min_number_per_label)

        self.target_names = self.trainObj.target_names
        self.n_categories = len(self.target_names)

        self.n_features_per_word = n_features_per_word
        wordEmbedPath = 'dataset/glove/glove_s{}.txt'.format(
            self.n_features_per_word)
        wordEmbedModel = KeyedVectors.load_word2vec_format(
            wordEmbedPath,
            unicode_errors="ignore")

        self.X_train_avg = self.sentence_to_avg(self.trainObj, wordEmbedModel)
        self.X_test_avg = self.sentence_to_avg(self.testObj, wordEmbedModel)

    def row_sentence_to_avg(self, row, wordEmbedModel, answer_list):
        """
        Converts a sentence (string) into a list of words (strings). Extracts
        the word2Vec representation of each word and averages its value into
        a single vector encoding the meaning of the sentence.
        """

        words = row.clean_text.lower().split()

        avg = np.zeros((self.n_features_per_word,))
        total = len(words)
        for word in words:
            try:
                avg += wordEmbedModel.word_vec(word)
            except KeyError:
                total -= 1
        if total != 0:
            avg = avg / total
        else:
            print("Clean text with no words in the embedding model for index {} .".format(row.name))
        answer_list.append(avg)

    def sentence_to_avg(self, dataObj, wordEmbedModel):
        X_avg = []
        dataObj.df.apply(self.row_sentence_to_avg, axis=1, args=[wordEmbedModel, X_avg])
        X_avg = np.array(X_avg)
        return X_avg

    def build_model(self):
        X_input = keras.layers.Input(shape=(self.n_features_per_word,))

        X = keras.layers.Dense(self.n_categories, name='fc')(X_input)
        X = keras.layers.Activation('softmax')(X)
        model = keras.models.Model(
            inputs=X_input,
            outputs=X,
            name='simple_model')

        return model

    def num_to_label(self, categoryNum):
        return self.target_names[categoryNum]

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

    def inspect_mispredictions(self, model, dataObj, X_avg, max_inspect_number):
        pred = model.predict(X_avg)
        mispredictions_count = 0
        for i in range(len(dataObj.target)):
            categoryNum = np.argmax(pred[i])
            if self.num_to_label(categoryNum) != dataObj.target.iloc[i]:
                print("\n\n Text:\n", dataObj.text.iloc[i])
                print('\nExpected category: {}\nPrediction: {}'.format(
                    dataObj.target.iloc[i],
                    self.num_to_label(categoryNum)))
                mispredictions_count += 1
                if mispredictions_count > max_inspect_number:
                    break
