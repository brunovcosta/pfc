import tensorflow as tf
from .word_embedding_model import WordEmbeddingModelKeras


class RNN(WordEmbeddingModelKeras):

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
        """
        Function creating the model's graph.

        Returns:
        model -- a model instance in Keras
        """
        raise NotImplementedError


class RNN2Layers(RNN):

    def _build_model(self):
        # it should be of dtype 'int32' (as it contains indices).
        sentence_indices = tf.keras.layers.Input(shape=(self.padded_length,), dtype='int32')
        embedding_layer = self.pretrained_embedding_layer()
        embeddings = embedding_layer(sentence_indices)

        X = tf.keras.layers.LSTM(128, return_sequences=True)(embeddings)
        X = tf.keras.layers.Dropout(0.5)(X)
        X = tf.keras.layers.LSTM(128, return_sequences=False)(X)
        X = tf.keras.layers.Dropout(0.5)(X)

        X = tf.keras.layers.Dense(self.n_categories)(X)
        X = tf.keras.layers.Activation('softmax')(X)

        model = tf.keras.models.Model(
            inputs=sentence_indices,
            outputs=X)

        return model


class RNNSimple(RNN):

    def _build_model(self):
        # it should be of dtype 'int32' (as it contains indices).
        sentence_indices = tf.keras.layers.Input(shape=(self.padded_length,), dtype='int32')
        embedding_layer = self.pretrained_embedding_layer()

        embeddings = embedding_layer(sentence_indices)
        X = tf.keras.layers.LSTM(128, return_sequences=False)(embeddings)

        X = tf.keras.layers.Dense(self.n_categories)(X)
        X = tf.keras.layers.Activation('softmax')(X)

        model = tf.keras.models.Model(
            inputs=sentence_indices,
            outputs=X)

        return model
