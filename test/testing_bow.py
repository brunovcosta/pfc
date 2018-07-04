import os
import sys

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.bag_of_words.classifiers import SVM


model = SVM(
    random_state=1,
    frac=0.001,
    dict_name="default.json",
    min_number_per_label=1000)
model.fit(save_metrics=False)
model.inspect_mispredictions('test', 3)
