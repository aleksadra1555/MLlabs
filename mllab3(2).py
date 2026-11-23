import itertools
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PolynomialRegression(BaseEstimator, RegressorMixin):

    def __init__(self, degree=2):
        self.degree = degree
        self.weight_matrix = None
        self.powers_ = None
        self.n_features_in_ = None


    @staticmethod
    def _powers(n_features, max_degree):
        """
        Все наборы показателей (e1, …, ep), сумма которых ≤ max_degree.
        Порядок — по возрастанию суммарной степени.
        Первый элемент — (0, …, 0) → свободный член.
        """
        powers = [] #списко кортежей, задающий мономы
        for d in range(0, max_degree + 1):
            if d == 0:
                powers.append(tuple([0] * n_features)) # соответствует w0
            else:
                for comb in itertools.combinations_with_replacement( #все сочетания длины d из индексов с повторениями 
                    range(n_features), d
                ):
                    exps = [0] * n_features # exps список показателей: то, сколько раз каждый признак входит в моном
                    for i in comb:
                        exps[i] += 1
                    powers.append(tuple(exps))
        return powers

    def _build_design(self, X):
        #Матрица Φ(X)
        n = X.shape[0]
        cols = []
        for exp in self.powers_:
            col = np.ones(n)
            for j, e in enumerate(exp):
                if e:
                    col = col * X[:, j] ** e
            cols.append(col)
        return np.column_stack(cols)

    #обучение
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X и y должны содержать одинаковое число наблюдений")

        self.powers_ = self._powers(X.shape[1], self.degree)
        Phi = self._build_design(X)

        # решаем нормальное уравнение
        self.weight_matrix = np.linalg.lstsq(Phi, y, rcond=None)[0].reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        return self


    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Phi = self._build_design(X)
        return (Phi @ self.weight_matrix).ravel()

    #Метрики
    def score(self, X, y):
        """R²"""
        return self.metrics(y, self.predict(X))["r2"]

    @staticmethod
    def metrics(y_true, y_pred):
        """mse, mae, r2"""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
        return {"mse": mse, "mae": mae, "r2": r2}
    
    def fit_grid(self, X, y, degrees=(1, 2, 3, 4, 5), cv=5,
                random_state=42, metric="r2"):
        """
        Подбирает degree по K-fold CV, затем переобучает на всех данных.
        metric: 'r2' | 'mse' | 'mae'
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        if cv < 2:
            raise ValueError("cv должно быть >= 2")
        if n < cv:
            raise ValueError("Мало наблюдений для заданного cv")

        rng = np.random.RandomState(random_state) #генератор случайных числел
        folds = np.array_split(rng.permutation(n), cv) #список из cv массивов индексов

        scores = {}
        for d in degrees:
            fold_scores = []
            for k in range(cv):
                test_idx  = folds[k]
                train_idx = np.concatenate(
                    [folds[j] for j in range(cv) if j != k]
                )
                m = PolynomialRegression(degree=d, rcond=self.rcond).fit(
                    X[train_idx], y[train_idx]
                )
                fold_scores.append(
                    m.metrics(y[test_idx], m.predict(X[test_idx]))[metric]
                )
            scores[d] = (float(np.mean(fold_scores)), float(np.std(fold_scores)))

        # R² максимизируем, ошибки — минимизируем
        best_d = (
            max(scores, key=lambda d: scores[d][0])
            if metric == "r2"
            else min(scores, key=lambda d: scores[d][0])
        )

        self.degree = best_d
        self.grid_scores_ = scores
        return self.fit(X, y)   # переобучение на всех данных