import numpy as np
import nltk
import sklearn
#from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from src.datasetAPI import RotaDosConcursos
from .base_model import BaseModel


class BagOfWords(BaseModel):

    def __init__(self, random_state=1, frac=1,
                 group_labels=False,
                 min_number_per_label=0):

        super(BagOfWords, self).__init__(
            random_state,
            frac,
            group_labels,
            min_number_per_label)

    def build_model(self):
        """
        Returns the model.
        """
        #Stemming
        stemmer = nltk.stem.RSLPStemmer()
        analyzer = sklearn.feature_extraction.text.CountVectorizer().build_analyzer()
        def stemmed(doc):
            return [stemmer.stem(w) for w in analyzer(doc)]

        model = Pipeline([
            ('bag of words', sklearn.feature_extraction.text.CountVectorizer(
                tokenizer=nltk.tokenize.word_tokenize,
                analyzer=stemmed,
                #ngram_range=(2,2),
                stop_words=nltk.corpus.stopwords.words('portuguese'),
                strip_accents='unicode')),
            ('tf-idf', sklearn.feature_extraction.text.TfidfTransformer()),
            ('clf-svm', sklearn.linear_model.SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=1e-3,
                n_iter=5,
                random_state=42))
        ])

        return model

    def execute_model(self):
        model = self.build_model()
        model.fit(self.trainObj.text, self.trainObj.target)
        predicted_svm = model.predict(self.testObj.text)
        mean_result = np.mean(predicted_svm == self.testObj.target)
        print(f"Mean result {mean_result}")
        self.inspect_mispredictions(model, self.testObj, self.testObj.text, 40)
