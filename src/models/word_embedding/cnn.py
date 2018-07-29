from tensorflow import keras
from .embedding_layer import EmbeddingLayer


"""
references

https://towardsdatascience.com/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-d2ee64b9dd0b
http://www.aclweb.org/anthology/D14-1181

https://github.com/bhaveshoswal/CNN-text-classification-keras
"""

class CNN(EmbeddingLayer):

    def _build_model(self):
        return keras.models.Sequential([
            self.pretrained_embedding_layer(),
            keras.layers.Conv1D(512, 3, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.MaxPooling1D(),
            keras.layers.Conv1D(512, 3, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.MaxPooling1D(),
            keras.layers.Conv1D(512, 3, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.MaxPooling1D(),
            keras.layers.Flatten(),
            keras.layers.Dense(self.n_categories, activation='softmax')
        ])
