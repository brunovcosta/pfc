import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from .word_embedding_model import WordEmbeddingModelKeras
from nltk.tokenize import word_tokenize

"""
references

https://towardsdatascience.com/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-d2ee64b9dd0b
http://www.aclweb.org/anthology/D14-1181
"""

class CNN(WordEmbeddingModelKeras):

    def _build_X_input(self, dataObj):
        print("build X input")
        train_data = dataObj
        train_data.df = train_data.df.iloc[:25]

        sentence_vectors = []
        for sentence in train_data.clean_text:
            try:
                words = [self.wordEmbedModel.word_vec(word) for word in word_tokenize(sentence.lower())]
                sentence_vectors.append(words)
            except KeyError:
                pass
            except AttributeError:
                print(sentence)

        self.max_sentence_size = max([len(sentence) for sentence in sentence_vectors])

        normalized_input = [np.concatenate((sentence,np.zeros((seld.max_sentence_size - len(sentence),self.n_features_per_word)))) for sentence in sentence_vectors]
        return normalized_input


    def _build_model(self):
        print("build model")
        # return tf.keras model
        batch_size = 32 # in each iteration, we consider 32 training examples at once
        num_epochs = 200 # we iterate 200 times over the entire training set
        kernel_size = 3 # we will use 3x3 kernels throughout
        pool_size = 2 # we will use 2x2 pooling throughout
        conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
        conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
        drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
        drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
        hidden_size = 512 # the FC layer will have 512 neurons

        dimensions = 50
        categories_count = 50



        input_layer = tf.keras.Input([self.max_sentence_size,dimensions])

        filter_layer_4_1 = tf.keras.layers.Conv1D(1, 4, activation='relu')(input_layer)
        filter_layer_4_2 = tf.keras.layers.Conv1D(1, 4, activation='relu')(input_layer)

        filter_layer_3_1 = tf.keras.layers.Conv1D(1, 3, activation='relu')(input_layer)
        filter_layer_3_2 = tf.keras.layers.Conv1D(1, 3, activation='relu')(input_layer)

        filter_layer_2_1 = tf.keras.layers.Conv1D(1, 2, activation='relu')(input_layer)
        filter_layer_2_2 = tf.keras.layers.Conv1D(1, 2, activation='relu')(input_layer)

        max_layer_4_1 = tf.keras.layers.MaxPooling1D(self.max_sentence_size - 4 + 1)(filter_layer_4_1)
        max_layer_4_2 = tf.keras.layers.MaxPooling1D(self.max_sentence_size - 4 + 1)(filter_layer_4_2)

        max_layer_3_1 = tf.keras.layers.MaxPooling1D(self.max_sentence_size - 3 + 1)(filter_layer_3_1)
        max_layer_3_2 = tf.keras.layers.MaxPooling1D(self.max_sentence_size - 3 + 1)(filter_layer_3_2)

        max_layer_2_1 = tf.keras.layers.MaxPooling1D(self.max_sentence_size - 2 + 1)(filter_layer_2_1)
        max_layer_2_2 = tf.keras.layers.MaxPooling1D(self.max_sentence_size - 2 + 1)(filter_layer_2_2)

        concat1max = tf.keras.layers.Concatenate(axis=1)([max_layer_4_1,
            max_layer_4_2,
            max_layer_3_1,
            max_layer_3_2,
            max_layer_2_1,
            max_layer_2_2])

        output_layer = tf.keras.layers.Dense(categories_count)(concat1max)


        return tf.keras.Model(input_layer,output_layer)
