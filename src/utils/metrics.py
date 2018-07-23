import tensorflow as tf
import numpy as np
import sklearn


class Metrics:

    def __init__(self):
        self.predict_probability = {}
        self.predict_label = {}
        self.target_one_hot = {}
        self.target_label = {}

    def accuracy_score(self, subset):
        return sklearn.metrics.accuracy_score(
            self.target_label[subset],
            self.predict_label[subset])

    def precision_score(self, subset):
        return sklearn.metrics.precision_score(
            self.target_label[subset],
            self.predict_label[subset],
            average='weighted')

    def recall_score(self, subset):
        return sklearn.metrics.recall_score(
            self.target_label[subset],
            self.predict_label[subset],
            average='weighted')

    def f1_score(self, subset):
        return sklearn.metrics.f1_score(
            self.target_label[subset],
            self.predict_label[subset],
            average='weighted')

    def roc_auc_score(self, subset):
        return sklearn.metrics.roc_auc_score(
            self.target_one_hot[subset],
            self.predict_probability[subset],
            average='weighted')

    def _build_parameters(self, subset, data, model=None):
        if self.predict_probability:
            self.predict_label[subset] = np.argmax(self.predict_probability[subset], axis=1)
        else:
            self.predict_label[subset] = model.predict(data[0])
        self.target_one_hot[subset] = np.asarray(data[1])
        self.target_label[subset] = np.argmax(self.target_one_hot[subset], axis=1)

    def get_metric_functions(self):
        prediction_metrics = [
            self.accuracy_score,
            self.precision_score,
            self.recall_score,
            self.f1_score,
        ]
        propability_prediction_metrics = [
            self.roc_auc_score
        ]
        metric_functions = prediction_metrics
        if self.predict_probability:
            metric_functions += propability_prediction_metrics
        return metric_functions


class MetricsBagOfWords(Metrics):

    def __init__(self, model, train_data, validation_data):
        super(MetricsBagOfWords, self).__init__()

        for subset, data in [('val', validation_data), ('train', train_data)]:
            try:
                self.predict_probability[subset] = np.asarray(model.predict_proba(data[0]))
            except AttributeError: # probability estimates are not available for loss='hinge'
                pass
            self._build_parameters(subset, data, model)

    def save_results(self, model_name):
        log_file = open(f'logs/txt_logs/{model_name}_metrics.txt', 'w')

        def calc_save(metric_func):
            for subset in ['val', 'train']:
                plot_name = metric_func.__name__
                metric_value = metric_func(subset)
                out_txt = f'{plot_name} - {subset} : {metric_value}\n'
                print(out_txt)
                log_file.write(out_txt)

        for metric_func in self.get_metric_functions():
            calc_save(metric_func)

        log_file.close()


class MetricsTensorboard(Metrics):

    def __init__(self, model, train_data, validation_data):
        super(MetricsTensorboard, self).__init__()
        self.loss = {}

        for subset, data in [('val', validation_data), ('train', train_data)]:
            self.predict_probability[subset] = np.asarray(model.predict(data[0]))
            self._build_parameters(subset, data)
            self.loss[subset] = model.evaluate(data[0], data[1])

    def write_metrics_to_tensorboard(self, writer, epoch):

        def write(value, plot_name, subset):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = plot_name
            writer[subset].add_summary(summary, epoch)

        for subset in ['val', 'train']:
            write(self.loss[subset], 'loss_end_epoch', subset)

        for metric_func in self.get_metric_functions():
            for subset in ['val', 'train']:
                plot_name = metric_func.__name__
                metric_value = metric_func(subset)
                write(metric_value, plot_name, subset)

        for subset in ['val', 'train']:
            writer[subset].flush()
