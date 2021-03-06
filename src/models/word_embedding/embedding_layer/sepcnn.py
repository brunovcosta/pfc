from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D

from .embedding_layer import EmbeddingLayer


"""
references

https://towardsdatascience.com/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-d2ee64b9dd0b
http://www.aclweb.org/anthology/D14-1181
sep-cnn: https://developers.google.com/machine-learning/guides/text-classification/step-4
https://github.com/bhaveshoswal/CNN-text-classification-keras
"""

class SepCNN(EmbeddingLayer):

    def _build_model(self):
        pretrained = self.pretrained_embedding_layer(mask_zero=True)
        return self.sepcnn_model(2,64,3,self.n_features_per_word,0.2,3,[self.max_text_len],self.n_categories,len(self.wordEmbedModel.vocab),True,False,pretrained)

    def sepcnn_model(self,
                     blocks,
                     filters,
                     kernel_size,
                     embedding_dim,
                     dropout_rate,
                     pool_size,
                     input_shape,
                     num_classes,
                     num_features,
                     use_pretrained_embedding=False,
                     is_embedding_trainable=False,
                     embedding_matrix=None):
        """Creates an instance of a separable CNN model.

        # Arguments
            blocks: int, number of pairs of sepCNN and pooling blocks in the model.
            filters: int, output dimension of the layers.
            kernel_size: int, length of the convolution window.
            embedding_dim: int, dimension of the embedding vectors.
            dropout_rate: float, percentage of input to drop at Dropout layers.
            pool_size: int, factor by which to downscale input at MaxPooling layer.
            input_shape: tuple, shape of input to the model.
            num_classes: int, number of output classes.
            num_features: int, number of words (embedding input dimension).
            use_pretrained_embedding: bool, true if pre-trained embedding is on.
            is_embedding_trainable: bool, true if embedding layer is trainable.
            embedding_matrix: dict, dictionary with embedding coefficients.

        # Returns
            A sepCNN model instance.
        """
        op_units, op_activation = self._get_last_layer_units_and_activation(num_classes)
        model = models.Sequential()
        model.add(self.pretrained_embedding_layer(mask_zero=False))
        for _ in range(blocks-1):
            model.add(Dropout(rate=dropout_rate))
            model.add(SeparableConv1D(filters=filters,
                                      kernel_size=kernel_size,
                                      activation='relu',
                                      bias_initializer='random_uniform',
                                      depthwise_initializer='random_uniform',
                                      padding='same'))
            model.add(SeparableConv1D(filters=filters,
                                      kernel_size=kernel_size,
                                      activation='relu',
                                      bias_initializer='random_uniform',
                                      depthwise_initializer='random_uniform',
                                      padding='same'))
            model.add(MaxPooling1D(pool_size=pool_size))

        model.add(SeparableConv1D(filters=filters * 2,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters * 2,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(op_units, activation=op_activation))
        return model

    def _get_last_layer_units_and_activation(self,num_classes):
        """Gets the # units and activation function for the last network layer.

        # Arguments
            num_classes: int, number of classes.

        # Returns
            units, activation values.
        """
        if num_classes == 2:
            activation = 'sigmoid'
            units = 1
        else:
            activation = 'softmax'
            units = num_classes
        return units, activation
