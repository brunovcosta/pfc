# Using BagOfWords as a model and NaiveBayes as the ML algorithm

import numpy as np
from src.datasetAPI import RotaDosConcursos
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

class BagOfWords:
    def __init__(self):
        self.trainObj = RotaDosConcursos(subset='train')
        self.testObj = RotaDosConcursos(subset='test')

        text_clf_svm = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', SGDClassifier(loss='hinge',penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
        ])
        
        _ = text_clf_svm.fit(self.trainObj.text, self.trainObj.target)
        predicted_svm = text_clf_svm.predict(self.testObj.text)
        self.mean_result = np.mean(predicted_svm == self.testObj.target)
        
        # X_trains_counts = count_vect.fit_transform(self.trainObj.loc)
        # X_train_counts.shape

        # nCategories = len(self.trainObj.target_names)

        # self.nFeaturesPerWord = 50
        # wordEmbedPath = 'dataset/glove/glove_s{}.txt'.format(
        #     self.nFeaturesPerWord)
        # wordEmbedModel = KeyedVectors.load_word2vec_format(
        #     wordEmbedPath,
        #     unicode_errors="ignore")

        # self.X_train_avg = self.vector_sentence_to_avg(wordEmbedModel)

        # model = self.simple_model(X_train_avg.shape, nCategories)
        # model.summary()

        # model.compile(
        #     loss='categorical_crossentropy',
        #     optimizer='adam',
        #     metrics=['accuracy'])

        # model.fit(
        #     X_train_avg,
        #     self.trainObj.target_one_hot,
        #     epochs=50,
        #     batch_size=32,
        #     shuffle=True)

        # loss, acc = model.evaluate(X_train_avg, self.trainObj.target_one_hot)
        # print("\nTrain accuracy = ", acc)

        # self.target_names = self.trainObj.target_names

        # pred = model.predict(X_train_avg)
        # for i in range(len(X_train_avg)):
        #     categoryNum = np.argmax(pred[i])
        #     if categoryNum != np.argmax(Y_oh_train[i]): #TODO
        #         print("\n\n Text:\n", X_train[i])
           
