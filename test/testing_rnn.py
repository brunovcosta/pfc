import os
import sys
import matplotlib.pyplot as plt

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.word_embedding import RNN_Simple


model = RNN_Simple(
    n_features_per_word=50,
    random_state=1,
    dict_name="default.json",
    min_number_per_label=0,
    frac=0.0001)

model.summary()

model.fit(save_metrics=False)

print(model.get_model().predict(model.get_X_input(model.trainObj)))

"""
plt.figure()
model.plot_confusion_matrix('test', title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
model.plot_confusion_matrix('test', normalize=True, title='Normalized confusion matrix')

plt.show()

model.inspect_mispredictions('test', 3)
"""
