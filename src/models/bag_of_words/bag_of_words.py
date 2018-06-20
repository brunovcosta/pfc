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

    def execute_model(self):
        model = self.build_model()
        model.fit(self.trainObj.text, self.trainObj.target)
        predicted = model.predict(self.testObj.text)
        mean_result = np.mean(predicted == self.testObj.target)
        print(f"Mean result {mean_result}")
        self.inspect_mispredictions(model, self.testObj, self.testObj.text, 40)
