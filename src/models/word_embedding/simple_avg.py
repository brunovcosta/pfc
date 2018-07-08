import numpy as np
import tensorflow as tf
from .word_embedding_model import WordEmbeddingModelKeras

class SimpleAvg(WordEmbeddingModelKeras):

    def row_sentence_to_avg(self, row, answer_list):
        """
        Extracts the word2Vec representation of each word and averages
        its value into a single vector encoding the meaning of the sentence.
        """
        avg = np.zeros((self.n_features_per_word,))
        total = len(row.splitted_text)
        for word in row.splitted_text:
            try:
                avg += self.wordEmbedModel.word_vec(word)
            except KeyError:
                total -= 1
        if total != 0:
            avg = avg / total
        else:
            print("Text with no words in the embedding model for index {} .".format(row.name))
        answer_list.append(avg)

    def _build_X_input(self, dataObj):
        X_avg = []
        dataObj.df.apply(self.row_sentence_to_avg, axis=1, args=[X_avg])
        X_avg = np.array(X_avg)
        return X_avg

    def _build_model(self):
        X_input = tf.keras.layers.Input(shape=(self.n_features_per_word,))
        X = tf.keras.layers.Dense(self.n_categories, name='fc')(X_input)
        X = tf.keras.layers.Activation('softmax')(X)
        model = tf.keras.models.Model(
            inputs=X_input,
            outputs=X,
            name='simple_model')

        return model
