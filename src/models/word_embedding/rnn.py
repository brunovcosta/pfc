import tensorflow as tf
import numpy as np
from .word_embedding_model import WordEmbeddingModelKeras


class RNN(WordEmbeddingModelKeras):

    def _row_sentences_to_indices(self, row, answer_list):
        """
        Converts an array of sentences (strings) into an array of indices corresponding
        to words in the sentences. The output shape should be such that it can
        be given to `Embedding()`
        """
        X_indices = np.zeros((self.max_text_len, ))

        for index, word in enumerate(row.splitted_text):
            try:
                X_indices[index] = self.wordEmbedModel.vocab[word].index
            except KeyError:
                X_indices[index] = self.wordEmbedModel.vocab['<unk>'].index

        answer_list.append(X_indices)

    def _build_X_input(self, dataObj):
        X_indices = []
        dataObj.df.apply(
            self._row_sentences_to_indices, axis=1,
            args=[X_indices])
        X_indices = np.array(X_indices)
        return X_indices

    def pretrained_embedding_layer(self):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe.

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """
        vocab_len = len(self.wordEmbedModel.vocab)

        embedding_layer = tf.keras.layers.Embedding(
            vocab_len,
            self.n_features_per_word)
        embedding_layer.trainable = False
        embedding_layer.build((None,))
        embedding_layer.set_weights([self.wordEmbedModel.vectors])

        return embedding_layer

    def _build_model(self):
        """
        Function creating the model's graph.

        Returns:
        model -- a model instance in Keras
        """
        raise NotImplementedError


class RNN2Layers(RNN):

    def _build_model(self):
        # Define sentence_indices as the input of the graph,
        # it should be of dtype 'int32' (as it contains indices).
        sentence_indices = tf.keras.layers.Input(shape=(self.max_text_len,), dtype='int32')

        # Create the embedding layer pretrained with GloVe Vectors
        embedding_layer = self.pretrained_embedding_layer()

        # Propagate sentence_indices through your embedding layer, you get back the embeddings
        embeddings = embedding_layer(sentence_indices)

        # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
        # The returned output should be a batch of sequences.
        X = tf.keras.layers.LSTM(128, return_sequences=True)(embeddings)
        # Add dropout with a probability of 0.5
        X = tf.keras.layers.Dropout(0.5)(X)
        # Propagate X trough another LSTM layer with 128-dimensional hidden state
        # The returned output should be a single hidden state, not a batch of sequences.
        X = tf.keras.layers.LSTM(128, return_sequences=False)(X)
        # Add dropout with a probability of 0.5
        X = tf.keras.layers.Dropout(0.5)(X)
        X = tf.keras.layers.Dense(self.n_categories)(X)
        # Add a softmax activation
        X = tf.keras.layers.Activation('softmax')(X)

        # Create Model instance which converts sentence_indices into X.
        model = tf.keras.models.Model(
            inputs=sentence_indices,
            outputs=X)

        return model


class RNNSimple(RNN):

    def _build_model(self):
        # Define sentence_indices as the input of the graph,
        # it should be of dtype 'int32' (as it contains indices).
        sentence_indices = tf.keras.layers.Input(shape=(self.max_text_len,), dtype='int32')

        # Create the embedding layer pretrained with GloVe Vectors
        embedding_layer = self.pretrained_embedding_layer()

        # Propagate sentence_indices through your embedding layer, you get back the embeddings
        embeddings = embedding_layer(sentence_indices)
        X = tf.keras.layers.LSTM(128, return_sequences=False)(embeddings)
        X = tf.keras.layers.Dense(self.n_categories)(X)
        # Add a softmax activation
        X = tf.keras.layers.Activation('softmax')(X)

        # Create Model instance which converts sentence_indices into X.
        model = tf.keras.models.Model(
            inputs=sentence_indices,
            outputs=X)

        return model
