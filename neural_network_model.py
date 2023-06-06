import pandas as pd


class NeuralNetworkModel:

    def __init__(self):
        self.history = None
        self.model = None

    def _preprocess(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self._preprocess(X_train)
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = self._preprocess(X)
        return self.model.predict(X)
        pass
