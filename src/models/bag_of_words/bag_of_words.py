import nltk
import sklearn
from ..base_model import BaseModel
from ...utils.metrics import MetricsBagOfWords


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
        print(f"fitting model {self}...")
        model = self.get_model()
        model.fit(
            self.get_X_input(self.data['train']),
            self.data['train'].target)
        if save_metrics:
            metrics = MetricsBagOfWords(
                model,
                train_data=(
                    self.get_X_input(self.data['train']),
                    self.data['train'].target_one_hot,
                ),
                validation_data=(
                    self.get_X_input(self.data['val']),
                    self.data['val'].target_one_hot
                )
            )
            metrics.save_results(str(self))
