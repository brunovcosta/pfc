import keras
import numpy as np
from .word_embedding_model import WordEmbeddingModelKeras


class RNN(WordEmbeddingModelKeras):

    def row_sentences_to_indices(self, row, max_len, answer_list):
        """
        Converts an array of sentences (strings) into an array of indices corresponding
        to words in the sentences. The output shape should be such that it can
        be given to `Embedding()`
        """

        X_indices = np.zeros((max_len, ))

        sentence_words = row.text.lower().split()

        for index, word in enumerate(sentence_words):
            try:
                X_indices[index] = self.wordEmbedModel.vocab[word].index
            except KeyError:
                X_indices[index] = self.wordEmbedModel.vocab['<unk>'].index

        answer_list.append(X_indices)

    def generate_X_input(self, dataObj):
        X_indices = []
        dataObj.df.apply(
            self.row_sentences_to_indices, axis=1,
            args=[dataObj.max_text_length("text"), X_indices])
        X_indices = np.array(X_indices)
        return X_indices

    def pretrained_embedding_layer(self):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe.

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """

        vocab_len = len(self.wordEmbedModel.vocab)

        embedding_layer = keras.layers.embeddings.Embedding(
            vocab_len,
            self.n_features_per_word,
            trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([self.wordEmbedModel.vectors])

        return embedding_layer

    def build_model(self):
        """
        Function creating the model's graph.

        Returns:
        model -- a model instance in Keras
        """

        # Define sentence_indices as the input of the graph, it should be of dtype 'int32' (as it contains indices).
        sentence_indices = keras.layers.Input(shape=(self.max_len,), dtype='int32')

        # Create the embedding layer pretrained with GloVe Vectors
        embedding_layer = self.pretrained_embedding_layer()

        # Propagate sentence_indices through your embedding layer, you get back the embeddings
        embeddings = embedding_layer(sentence_indices)

        # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
        # The returned output should be a batch of sequences.
        X = keras.layers.LSTM(128, return_sequences=True)(embeddings)
        # Add dropout with a probability of 0.5
        X = keras.layers.Dropout(0.5)(X)
        # Propagate X trough another LSTM layer with 128-dimensional hidden state
        # The returned output should be a single hidden state, not a batch of sequences.
        X = keras.layers.LSTM(128, return_sequences=False)(X)
        # Add dropout with a probability of 0.5
        X = keras.layers.Dropout(0.5)(X)
        X = keras.layers.Dense(self.n_categories)(X)
        # Add a softmax activation
        X = keras.layers.Activation('softmax')(X)

        # Create Model instance which converts sentence_indices into X.
        model = keras.models.Model(
            inputs=sentence_indices,
            outputs=X)

        return model
