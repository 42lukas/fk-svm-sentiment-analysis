from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

class SVM_Classifier:
    def __init__(self, C=1.0, max_iter=2000):
        self.model = LinearSVC(C=C, dual=False, max_iter=max_iter)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)