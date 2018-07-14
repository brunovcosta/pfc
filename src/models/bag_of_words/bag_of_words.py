import numpy as np
import nltk
import sklearn
from ..base_model import BaseModel


class BagOfWords(BaseModel):

    def get_initial_pipeline_steps(self):
        #Stemming
        stemmer = nltk.stem.RSLPStemmer()
        analyzer = sklearn.feature_extraction.text.CountVectorizer().build_analyzer()
        def stemmed(doc):
            return [stemmer.stem(w) for w in analyzer(doc)]

        steps = [
            ('bag of words', sklearn.feature_extraction.text.CountVectorizer(
                tokenizer=nltk.tokenize.word_tokenize,
                analyzer=stemmed,
                #ngram_range=(2,2),
                stop_words=nltk.corpus.stopwords.words('portuguese'),
                strip_accents='unicode')),
            ('tf-idf', sklearn.feature_extraction.text.TfidfTransformer())
        ]

        return steps

    def _build_X_input(self, dataObj):
        return dataObj.text

    def fit(self, save_metrics=False, save_checkpoints=False):
        model = self.get_model()
        model.fit(
            self.get_X_input(self.trainObj),
            self.trainObj.target)
        if save_metrics:
            predicted = model.predict(self.get_X_input(self.testObj))
            mean_result = np.mean(predicted == self.testObj.target)
            print(f"Mean result {mean_result}")
