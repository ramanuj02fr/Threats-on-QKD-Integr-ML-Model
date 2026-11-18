"""SVM with RBF kernel (scores = decision_function; no probability calibration, which would be ~5x slower)."""
from sklearn.svm import SVC
from model_common import run


def make_model():
    return SVC(kernel="rbf", cache_size=1000)


if __name__ == "__main__":
    run("SVM", make_model)
