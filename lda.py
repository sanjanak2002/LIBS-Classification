# lda.py

import numpy as np

class LDAClassifier:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.means = [np.mean(X_train[y_train == c], axis=0) for c in self.classes]

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - mean) for mean in self.means]
        return self.classes[np.argmin(distances)]
