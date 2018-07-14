from .word_embedding_model import WordEmbeddingModelKeras
import numpy as np
import tensorflow as tf
Input = tf.keras.layers.Input
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Embedding = tf.keras.layers.Embedding
Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
Reshape = tf.keras.layers.Reshape
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
Permute = tf.keras.layers.Permute
Concatenate = tf.keras.layers.Concatenate
ModelCheckPoint = tf.keras.callbacks.ModelCheckpoint
Adam = tf.keras.optimizers.Adam
Model = tf.keras.models.Model


"""
references

https://towardsdatascience.com/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-d2ee64b9dd0b
http://www.aclweb.org/anthology/D14-1181

https://github.com/bhaveshoswal/CNN-text-classification-keras
"""

class CNN(WordEmbeddingModelKeras):

    def _build_X_input(self, dataObj):
        print("build X input")
        train_data = dataObj

        sentence_vectors = []
        for sentence in train_data.splitted_text:
            words_embedding = []
            for word in sentence:
                try:
                    words_embedding.append(self.wordEmbedModel.word_vec(word))
                except KeyError:
                    words_embedding.append(self.wordEmbedModel.word_vec('<unk>'))
                except AttributeError:
                    print(sentence)
            sentence_vectors.append(words_embedding)

        normalized_input = [np.concatenate((sentence,np.zeros((self.max_text_len - len(sentence),self.n_features_per_word)))) for sentence in sentence_vectors]
        return np.array(normalized_input)

    def _build_model(self):
        sequence_length = self.max_text_len
        embedding_dim = self.n_features_per_word
        filter_sizes = [3, 4, 5]
        num_filters = 512
        drop = 0.5

        inputs = Input(shape=(sequence_length,embedding_dim))

        reshape = Reshape((sequence_length, embedding_dim,1))(inputs)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=self.n_categories, activation='softmax')(dropout)

        return Model(inputs=inputs, outputs=output)
