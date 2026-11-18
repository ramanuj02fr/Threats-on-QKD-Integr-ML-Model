"""
Neural Network (MLP): 15 inputs -> Dense 64 (ReLU) -> Dense 32 (ReLU) -> 1 output (sigmoid).
Adam, batch size 512, early stopping on a 15 % validation split (patience 5), max 100 epochs.

scikit-learn's MLPClassifier is used so the script runs on any Python version without TensorFlow.
Difference to the Keras notebook used for the reported numbers: no Dropout layers (sklearn has none).
"""
from sklearn.neural_network import MLPClassifier
from model_common import run, SEED


def make_model():
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", batch_size=512,
                         max_iter=100, early_stopping=True, validation_fraction=0.15, n_iter_no_change=5,
                         random_state=SEED)


if __name__ == "__main__":
    run("Neural Network", make_model)
