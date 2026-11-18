"""Random Forest (100 trees, all CPU cores)."""
from sklearn.ensemble import RandomForestClassifier
from model_common import run, SEED


def make_model():
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)


if __name__ == "__main__":
    run("Random Forest", make_model)
