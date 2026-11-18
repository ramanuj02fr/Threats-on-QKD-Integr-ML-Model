"""Logistic Regression (linear model, features standardized)."""
from sklearn.linear_model import LogisticRegression
from model_common import run


def make_model():
    return LogisticRegression(max_iter=1000)


if __name__ == "__main__":
    run("Logistic Regression", make_model)
