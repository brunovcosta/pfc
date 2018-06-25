import numpy as np
import tensorflow as tf
import nltk
from .word_embedding_model import WordEmbeddingModelKeras

class SimpleAvg(WordEmbeddingModelKeras):

    def row_sentence_to_avg(self, row, answer_list):
        """
        Converts a sentence (string) into a list of words (strings). Extracts
        the word2Vec representation of each word and averages its value into
        a single vector encoding the meaning of the sentence.
        """

        words = nltk.tokenize.word_tokenize(row.clean_text.lower())

        avg = np.zeros((self.n_features_per_word,))
        total = len(words)
        for word in words:
            try:
                avg += self.wordEmbedModel.word_vec(word)
            except KeyError:
                total -= 1
        if total != 0:
            avg = avg / total
        else:
            print("Clean text with no words in the embedding model for index {} .".format(row.name))
        answer_list.append(avg)

    def generate_X_input(self, dataObj):
        X_avg = []
        dataObj.df.apply(self.row_sentence_to_avg, axis=1, args=[X_avg])
        X_avg = np.array(X_avg)
        return X_avg

    def build_model(self):
        X_input = tf.keras.layers.Input(shape=(self.n_features_per_word,))
        X = tf.keras.layers.Dense(self.n_categories, name='fc')(X_input)
        X = tf.keras.layers.Activation('softmax')(X)
        model = tf.keras.models.Model(
            inputs=X_input,
            outputs=X,
            name='simple_model')

        return model
