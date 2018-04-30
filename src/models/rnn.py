import keras
import numpy as np
from gensim.models import KeyedVectors
from ..datasetAPI import RotaDosConcursos


class RNN:

    def __init__(self, n_features_per_word=50, random_state=1, frac=1,
                 group_labels=False, min_number_per_label=0):
        self.trainObj = RotaDosConcursos(
            subset='train',
            random_state=random_state,
            group_labels=group_labels,
            min_number_per_label=min_number_per_label)
        self.testObj = RotaDosConcursos(
            subset='test',
            random_state=random_state,
            group_labels=group_labels,
            min_number_per_label=min_number_per_label)

        self.max_len = self.trainObj.max_text_length("text")
        self.target_names = self.trainObj.target_names
        self.n_categories = len(self.target_names)

        self.n_features_per_word = n_features_per_word
        wordEmbedPath = 'dataset/glove/glove_s{}.txt'.format(
            self.n_features_per_word)
        self.wordEmbedModel = KeyedVectors.load_word2vec_format(
            wordEmbedPath,
            unicode_errors="ignore")

        self.train_indices = self.sentences_to_indices(self.trainObj)
        self.test_indices = self.sentences_to_indices(self.testObj)

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

    def sentences_to_indices(self, dataObj):
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

    def num_to_label(self, categoryNum):
        return self.target_names[categoryNum]

    def execute_model(self):
        model = self.build_model()
        model.summary()

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        model.fit(
            self.train_indices,
            self.trainObj.target_one_hot,
            epochs=9,
            batch_size=32,
            shuffle=True)

        loss, acc = model.evaluate(self.train_indices, self.trainObj.target_one_hot)
        print("\nTrain accuracy = ", acc)

        loss, acc = model.evaluate(self.test_indices, self.testObj.target_one_hot)
        print("\nTest accuracy = ", acc)

        #self.inspect_mispredictions(model, self.trainObj, self.train_indices, 40)
        self.inspect_mispredictions(model, self.testObj, self.test_indices, 40)

    def inspect_mispredictions(self, model, dataObj, X_indices, max_inspect_number):
        pred = model.predict(X_indices)
        mispredictions = 0
        for i in range(len(dataObj.target)):
            categoryNum = np.argmax(pred[i])
            if self.num_to_label(categoryNum) != dataObj.target.iloc[i]:
                print("\n\n Text:\n", dataObj.text.iloc[i])
                print('\nExpected category: {}\nPrediction: {}'.format(
                    dataObj.target.iloc[i],
                    self.num_to_label(categoryNum)))
                mispredictions += 1
                if mispredictions > max_inspect_number:
                    break
