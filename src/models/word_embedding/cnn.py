from .word_embedding_model import WordEmbeddingModelKeras
import numpy as np
import tensorflow as tf
from tensorflow import keras


"""
references

https://towardsdatascience.com/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-d2ee64b9dd0b
http://www.aclweb.org/anthology/D14-1181

https://github.com/bhaveshoswal/CNN-text-classification-keras
"""

class CNN(WordEmbeddingModelKeras):

    def _row_sentences_to_indices(self, row, answer_list):
        """
        Converts a list of strings into a list of indices corresponding
        to words in the word embedding model.
        """
        X_indices = []
        for word in row.splitted_text:
            try:
                word_index = self.wordEmbedModel.vocab[word].index
            except KeyError:
                word_index = self.wordEmbedModel.vocab['<unk>'].index
            X_indices.append(word_index)

        answer_list.append(X_indices)

    def _build_X_input(self, dataObj):
        X_indices = []

        dataObj.df.apply(
            self._row_sentences_to_indices, axis=1,
            args=[X_indices])

        X_indices = tf.keras.preprocessing.sequence.pad_sequences(
            X_indices,
            maxlen=self.padded_length,
            padding='pre',
            truncating='post')
        return X_indices

    def pretrained_embedding_layer(self):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe.

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """
        vocab_len = len(self.wordEmbedModel.vocab)

        embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_len,
            output_dim=self.n_features_per_word,
            input_length=self.padded_length)
        embedding_layer.trainable = False
        embedding_layer.build((None,))
        embedding_layer.set_weights([self.wordEmbedModel.vectors])

        return embedding_layer

    def _build_model(self):
        return keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.padded_length,), dtype='int32'),
            self.pretrained_embedding_layer(),
            keras.layers.Conv2D(10, (3,self.n_features_per_word), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(10, (3,self.n_features_per_word), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(self.n_categories, activation='softmax')
        ])
