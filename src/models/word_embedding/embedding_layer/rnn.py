import tensorflow as tf
from .embedding_layer import EmbeddingLayer


class StackedRNN(EmbeddingLayer):

    def _build_model(self):
        # it should be of dtype 'int32' (as it contains indices).
        sentence_indices = tf.keras.layers.Input(shape=(self.padded_length,), dtype='int32')
        embedding_layer = self.pretrained_embedding_layer(mask_zero=True)
        embeddings = embedding_layer(sentence_indices)

        X = tf.keras.layers.LSTM(2*self.n_features_per_word, return_sequences=True)(embeddings)
        X = tf.keras.layers.Dropout(0.5)(X)
        X = tf.keras.layers.LSTM(2*self.n_features_per_word, return_sequences=False)(X)
        X = tf.keras.layers.Dropout(0.5)(X)

        X = tf.keras.layers.Dense(self.n_categories)(X)
        X = tf.keras.layers.Activation('softmax')(X)

        model = tf.keras.models.Model(
            inputs=sentence_indices,
            outputs=X)

        return model


class RNN(EmbeddingLayer):

    def _build_model(self):
        # it should be of dtype 'int32' (as it contains indices).
        sentence_indices = tf.keras.layers.Input(shape=(self.padded_length,), dtype='int32')
        embedding_layer = self.pretrained_embedding_layer(mask_zero=True)

        embeddings = embedding_layer(sentence_indices)
        X = tf.keras.layers.LSTM(self.n_features_per_word, return_sequences=False)(embeddings)

        X = tf.keras.layers.Dense(self.n_categories)(X)
        X = tf.keras.layers.Activation('softmax')(X)

        model = tf.keras.models.Model(
            inputs=sentence_indices,
            outputs=X)

        return model


class BidirectionalGRUConv(EmbeddingLayer):

    def _build_model(self):
        sentence_indices = tf.keras.layers.Input(shape=(self.padded_length,), dtype='int32')
        embedding_layer = self.pretrained_embedding_layer(mask_zero=False)

        embeddings = embedding_layer(sentence_indices)
        X = tf.keras.layers.SpatialDropout1D(0.2)(embeddings)
        X = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                128,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1))(X)
        X = tf.keras.layers.Conv1D(
            64,
            kernel_size=3,
            padding="valid",
            kernel_initializer="glorot_uniform")(X)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(X)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(X)
        X = tf.keras.layers.concatenate([avg_pool, max_pool])

        X = tf.keras.layers.Dense(self.n_categories)(X)
        X = tf.keras.layers.Activation('softmax')(X)

        model = tf.keras.models.Model(
            inputs=sentence_indices,
            outputs=X)
        return model


class ConvLSTM(EmbeddingLayer):

    def _build_model(self):
        sentence_indices = tf.keras.layers.Input(shape=(self.padded_length,), dtype='int32')
        embedding_layer = self.pretrained_embedding_layer(mask_zero=False)

        embeddings = embedding_layer(sentence_indices)
        X = tf.keras.layers.Dropout(0.25)(embeddings)
        X = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=5,
            padding='valid',
            activation='relu',
            strides=1)(X)
        X = tf.keras.layers.MaxPooling1D(pool_size=4)(X)
        X = tf.keras.layers.LSTM(70)(X)
        X = tf.keras.layers.Dense(self.n_categories)(X)
        X = tf.keras.layers.Activation('softmax')(X)

        model = tf.keras.models.Model(
            inputs=sentence_indices,
            outputs=X)
        return model
