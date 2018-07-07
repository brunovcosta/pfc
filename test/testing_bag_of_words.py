import os
import sys
print("WAHEY, it just began")
module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))
print("WAHEY, it just started")

from src.models.bag_of_words.classifiers import SVM
print("WAHEY, it just almost ending")


model = SVM(
    random_state=1,
    frac=1,
    dict_name="default.json",
    min_number_per_label=1000)
model.fit(save_metrics=True)
model.inspect_mispredictions('test', 3)
print("WAHEY, it just end")
