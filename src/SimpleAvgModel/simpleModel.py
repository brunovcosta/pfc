import numpy as np
import keras
from gensim.models import KeyedVectors


def convert_to_one_hot(Y, nCategories):
    pass


def loadData(path):
    pass


fake_path = "BaixarDataset"
fake_nCategories = 40

X_train, Y_train = loadData(fake_path)
X_test, Y_test = loadData(fake_path)

Y_oh_train = convert_to_one_hot(Y_train, fake_nCategories)
Y_oh_test = convert_to_one_hot(Y_test, fake_nCategories)


def max_word_length(X):
    splittedXlen = map(lambda x: len(x.split()), X)
    return max(splittedXlen)


nFeaturesPerWord = 300
wordEmbedPath = '../../dataset/glove/glove_s{}.txt'.format(str(nFeaturesPerWord))
wordEmbedModel = KeyedVectors.load_word2vec_format(wordEmbedPath, unicode_errors="ignore")

wordEmbedModel.word_vec('casa')


def sentence_to_avg(sentence, wordEmbedModel):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the word2Vec representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array
    """

    words = sentence.lower().split()    # !!!!!!!!Revisar lower para palavras do Pt

    nFeaturesPerWord = len(wordEmbedModel.word_vec('casa'))

    avg = np.zeros((nFeaturesPerWord,))

    for w in words:
        avg += wordEmbedModel.word_vec(w)  # !!!!!!!!!!!Adicionar <UNK> ao vocabulario (media de todas as palavras?)
    avg = avg / len(words)

    return avg


def vector_sentence_to_avg(X_sentences, wordEmbedModel):
    X = []

    for sentence in X_sentences:
        X.append(sentence_to_avg(sentence, wordEmbedModel))

    return X


def simple_model(input_shape, nCategories):
    X_input = keras.layers.Input(input_shape)

    X = keras.layers.Dense(nCategories, name='fc')(X_input)
    X = keras.layers.Activation('softmax')(X)
    model = keras.models.Model(inputs=X_input, outputs=X, name='simple_model')

    return model


X_train_avg = vector_sentence_to_avg(X_train, fake_nCategories)

model = simple_model(X_train_avg.shape, fake_nCategories)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_avg, Y_oh_train, epochs=50, batch_size=32, shuffle=True)

loss, acc = model.evaluate(X_train_avg, Y_oh_test)
print("\nTrain accuracy = ", acc)


def label_to_category(categoryNum):
    pass


pred = model.predict(X_train_avg)
for i in range(len(X_train_avg)):
    categoryNum = np.argmax(pred[i])
    if(categoryNum != np.argmax(Y_oh_train[i])):
        print("\n\n Text:\n", X_train[i])
        print('\nExpected category:' + Y_train[i] + ' prediction: ' + label_to_category(categoryNum).strip())
