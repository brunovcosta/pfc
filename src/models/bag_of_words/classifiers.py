import sklearn
from sklearn.pipeline import Pipeline
from ..bag_of_words import BagOfWords


class SVM(BagOfWords):

    def build_model(self):

        steps = self.get_initial_pipeline_steps()
        steps.append(
            ('clf-svm', sklearn.linear_model.SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=1e-3,
                n_iter=5,
                random_state=42))
        )
        model = Pipeline(steps)

        return model


class NB(BagOfWords):

    def build_model(self):

        steps = self.get_initial_pipeline_steps()
        steps.append(
            ('clf-naive_bayes', sklearn.naive_bayes.MultinomialNB())
        )
        model = Pipeline(steps)

        return model
