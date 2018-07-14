import os
import sys
import matplotlib.pyplot as plt

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.word_embedding import CNN


model = CNN(
    n_features_per_word=50,
    random_state=1,
    dict_name="default.json",
    min_number_per_label=10000,
    frac=1)

model.fit(save_metrics=False)

plt.figure()
model.plot_confusion_matrix('test', title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
model.plot_confusion_matrix('test', normalize=True, title='Normalized confusion matrix')

plt.show()

model.inspect_mispredictions('test', 3)

model.summary()