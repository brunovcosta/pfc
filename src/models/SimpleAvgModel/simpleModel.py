import numpy as np
import keras
from gensim.models import KeyedVectors
import sys
sys.path.insert(0, '../../datasetAPI')
import datasetAPI


trainObj = datasetAPI.RotaDosConcursos(subset='train')
testObj = datasetAPI.RotaDosConcursos(subset='test')

nCategories = len(trainObj.target_names)


def max_word_length(X):
    splittedXlen = map(lambda x: len(x.split()), X)
    return max(splittedXlen)


nFeaturesPerWord = 50
wordEmbedPath = '../../../dataset/glove/glove_s{}.txt'.format(str(nFeaturesPerWord))
wordEmbedModel = KeyedVectors.load_word2vec_format(wordEmbedPath, unicode_errors="ignore")

wordEmbedModel.word_vec('casa')


def sentence_to_avg(sentence, wordEmbedModel):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the word2Vec representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array
    """

    words = sentence.lower().split()

    nFeaturesPerWord = len(wordEmbedModel.word_vec('casa'))

    avg = np.zeros((nFeaturesPerWord,))

    total = len(words)
    for w in words:
        try:
            avg += wordEmbedModel.word_vec(w)
        except KeyError:
            total -= 1
    avg = avg / total

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


X_train_avg = vector_sentence_to_avg(trainObj.text, wordEmbedModel)

model = simple_model(X_train_avg.shape, nCategories)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_avg, trainObj.target_one_hot, epochs=50, batch_size=32, shuffle=True)

loss, acc = model.evaluate(X_train_avg, trainObj.target_one_hot)
print("\nTrain accuracy = ", acc)


target_names = trainObj.target_names
def label_to_category(target_names, categoryNum):
    return target_names[categoryNum]


pred = model.predict(X_train_avg)
for i in range(len(X_train_avg)):
    categoryNum = np.argmax(pred[i])
    if categoryNum != np.argmax(Y_oh_train[i]):
        print("\n\n Text:\n", X_train[i])
        print('\nExpected category:' + trainObj.target.iloc[i] + ' prediction: ' + label_to_category(categoryNum).strip())
