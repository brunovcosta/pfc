import numpy as np
import keras
from gensim.models import KeyedVectors
from ..datasetAPI import RotaDosConcursos

class CNN:
    def __init__(self):
        train_data = RotaDosConcursos(subset='train')
        test_data = RotaDosConcursos(subset='test')

        train_categories_len = len(self.trainObj.target_names)

        self.nFeaturesPerWord = 50
        word_embed_path = 'dataset/glove/glove_s{}.txt'.format(
            self.nFeaturesPerWord)
        word_embed_model = KeyedVectors.load_word2vec_format(
            word_embed_path,
            unicode_errors="ignore")

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
