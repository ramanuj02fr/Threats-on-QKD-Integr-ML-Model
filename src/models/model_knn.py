"""K-Nearest Neighbours (k = 5)."""
from sklearn.neighbors import KNeighborsClassifier
from model_common import run


def make_model():
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


if __name__ == "__main__":
    run("KNN", make_model)
