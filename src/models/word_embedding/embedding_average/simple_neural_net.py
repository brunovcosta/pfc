from .embedding_average import EmbeddingAverage
import tensorflow as tf

class SimpleNeuralNet(EmbeddingAverage):

    def _build_model(self):
        X_input = tf.keras.layers.Input(shape=(self.n_features_per_word,))
        X = tf.keras.layers.Dense(self.n_categories, name='fc')(X_input)
        X = tf.keras.layers.Activation('softmax')(X)
        model = tf.keras.models.Model(
            inputs=X_input,
            outputs=X,
            name='simple_model')

        return model
