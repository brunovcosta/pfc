import tensorflow as tf
import numpy as np
from ..word_embedding_model import WordEmbeddingModelKeras


class EmbeddingLayer(WordEmbeddingModelKeras):

    def _row_sentences_to_indices(self, row, answer_list):
        """
        Converts a list of strings into a list of indices corresponding
        to words in the word embedding model.
        """
        X_indices = []
        for word in row.splitted_text:
            try:
                word_index = self.wordEmbedModel.vocab[word].index + 1
            except KeyError:
                word_index = self.wordEmbedModel.vocab['<unk>'].index + 1
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

    def pretrained_embedding_layer(self, mask_zero=False):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe.

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """
        vocab_len = len(self.wordEmbedModel.vocab)

        embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_len+1,
            output_dim=self.n_features_per_word,
            input_length=self.padded_length,
            mask_zero=mask_zero)
        embedding_layer.trainable = False
        embedding_layer.build((None,))

        weights = self.wordEmbedModel.vectors
        weights = np.append(
            np.zeros((1, self.n_features_per_word)),
            weights,
            axis=0
        )
        embedding_layer.set_weights([weights])

        return embedding_layer

    def _build_model(self):
        """
        Function creating the model's graph.

        Returns:
        model -- a model instance in Keras
        """
        raise NotImplementedError
