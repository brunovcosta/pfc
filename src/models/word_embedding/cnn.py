import numpy as np
import tensorflow as tf
from .word_embedding_model import WordEmbeddingModelKeras

"""
references

https://towardsdatascience.com/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-d2ee64b9dd0b
http://www.aclweb.org/anthology/D14-1181
"""

class CNN(WordEmbeddingModelKeras):

    def _build_X_input(self, dataObj):
        pass
        # return np.array

    def _build_model(self):
        pass    
        # return keras model
        """
        word_vectors = [word_embed_model[word] for word in train_data.text]

        batch_size = 32 # in each iteration, we consider 32 training examples at once
        num_epochs = 200 # we iterate 200 times over the entire training set
        kernel_size = 3 # we will use 3x3 kernels throughout
        pool_size = 2 # we will use 2x2 pooling throughout
        conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
        conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
        drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
        drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
        hidden_size = 512 # the FC layer will have 512 neurons
        """

        """
        import keras
        import numpy as np
        from src.datasetAPI import RotaDosConcursos
        from gensim.models import KeyedVectors
        from nltk.tokenize import word_tokenize

        dimensions = 50
        categories_count = 50

        train_data = RotaDosConcursos(subset='train')
        train_data.df = train_data.df.iloc[:25]
        #test_data = RotaDosConcursos(subset='test')

        train_categories_len = len(train_data.target_names)

        word_embed_path = 'dataset/glove/glove_s%d.txt'%dimensions
        word_embed_model = KeyedVectors.load_word2vec_format(
            word_embed_path,
            unicode_errors="ignore")

        sentence_vectors = []
        for sentence in train_data.clean_text:
            try:
                sentence_vectors.append([word_embed_model[word] for word in word_tokenize(sentence.lower())])
            except KeyError:
                pass
            except AttributeError:
                print(sentence)

        max_sentence_size = max([len(sentence) for sentence in sentence_vectors])

        normalized_input = [np.concatenate((sentence,np.zeros((max_sentence_size - len(sentence),dimensions)))) for sentence in sentence_vectors]

        input_layer = keras.Input([max_sentence_size,dimensions])

        filter_layer_4_1 = keras.layers.Conv1D(1, 4, activation='relu')(input_layer)
        filter_layer_4_2 = keras.layers.Conv1D(1, 4, activation='relu')(input_layer)

        filter_layer_3_1 = keras.layers.Conv1D(1, 3, activation='relu')(input_layer)
        filter_layer_3_2 = keras.layers.Conv1D(1, 3, activation='relu')(input_layer)

        filter_layer_2_1 = keras.layers.Conv1D(1, 2, activation='relu')(input_layer)
        filter_layer_2_2 = keras.layers.Conv1D(1, 2, activation='relu')(input_layer)

        max_layer_4_1 = keras.layers.MaxPooling1D(max_sentence_size - 4 + 1)(filter_layer_4_1)
        max_layer_4_2 = keras.layers.MaxPooling1D(max_sentence_size - 4 + 1)(filter_layer_4_2)

        max_layer_3_1 = keras.layers.MaxPooling1D(max_sentence_size - 3 + 1)(filter_layer_3_1)
        max_layer_3_2 = keras.layers.MaxPooling1D(max_sentence_size - 3 + 1)(filter_layer_3_2)

        max_layer_2_1 = keras.layers.MaxPooling1D(max_sentence_size - 2 + 1)(filter_layer_2_1)
        max_layer_2_2 = keras.layers.MaxPooling1D(max_sentence_size - 2 + 1)(filter_layer_2_2)

        concat1max = keras.layers.Concatenate(axis=1)([max_layer_4_1,
                                                    max_layer_4_2,
                                                    max_layer_3_1,
                                                    max_layer_3_2,
                                                    max_layer_2_1,
                                                    max_layer_2_2])

        output_layer = keras.layers.Dense(categories_count)(concat1max)


        model = keras.Model(input_layer,output_layer)

        model.summary()

        """
