"""Gradient Boosting (scikit-learn defaults: 100 trees, depth 3, learning rate 0.1)."""
from sklearn.ensemble import GradientBoostingClassifier
from model_common import run, SEED


def make_model():
    return GradientBoostingClassifier(random_state=SEED)


if __name__ == "__main__":
    run("Gradient Boosting", make_model)
