import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from src.datasetAPI import RotaDosConcursos

class BagOfWords:
    """
	mean_result : return the mean result of the SGDClassifier with the test subset
	"""
    def __init__(self):
        self.train_obj = RotaDosConcursos(subset='train')
        self.test_obj = RotaDosConcursos(subset='test')

        text_clf_svm = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
        ])

        _ = text_clf_svm.fit(self.train_obj.text, self.train_obj.target)
        predicted_svm = text_clf_svm.predict(self.test_obj.text)
        self.mean_result = np.mean(predicted_svm == self.test_obj.target)
