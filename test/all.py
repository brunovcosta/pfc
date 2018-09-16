import os
import sys

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.word_embedding.embedding_average import SimpleNeuralNet
from src.models.word_embedding.embedding_layer import CNN
from src.models.word_embedding.embedding_layer import RNN
from src.models.word_embedding.embedding_layer import ConvLSTM
from src.models.word_embedding.embedding_layer import SepCNN
from src.models.word_embedding.embedding_layer import StackedRNN
from src.models.bag_of_words.classifiers import NB
from src.models.bag_of_words.classifiers import SVM
from src.datasetAPI import RotaDosConcursos

print("loading dataset...")
rota_dos_concursos = RotaDosConcursos(
    frac=1,
    random_state=1,
    dict_name="default.json",
    min_number_per_label=10000)

models=[
    NB(rota_dos_concursos),

    SVM(rota_dos_concursos),

    SimpleNeuralNet(
        rota_dos_concursos,
        n_features_per_word=int(sys.argv[1]),
        hyperparameters_file="default"),

    CNN(
        rota_dos_concursos,
        n_features_per_word=int(sys.argv[1]),
        hyperparameters_file="default"),

    ConvLSTM(
        rota_dos_concursos,
        n_features_per_word=int(sys.argv[1]),
        hyperparameters_file="default"),

    RNN(
        rota_dos_concursos,
        n_features_per_word=int(sys.argv[1]),
        hyperparameters_file="default"),

    SepCNN(
        rota_dos_concursos,
        n_features_per_word=int(sys.argv[1]),
        hyperparameters_file="default"),

    StackedRNN(
        rota_dos_concursos,
        n_features_per_word=int(sys.argv[1]),
        hyperparameters_file="default")

]

for model in models:
    model.summary(save_summary=True)

    model.fit(
        save_metrics=True,
        save_checkpoints=True)

    model.save_plots()
    model.inspect_mispredictions('val', 10)