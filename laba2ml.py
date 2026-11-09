import numpy as np

class ManualLinearRegression:


    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept  #флаг наличия w0
        self.coef_ = None  #веса без w0
        self.intercept_ = None  #w0
        self.n_features_in_ = None  #количество признаков

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        X = self._as_2d(X)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X и y содержат разное число объектов")

        self.n_features_in_ = X.shape[1]
        Xb = self._add_intercept(X) if self.fit_intercept else X

        # w = (X^T X)^{-1} X^T y
        gram = Xb.T @ Xb
        rhs = Xb.T @ y

        w = np.linalg.solve(gram, rhs)

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(self, X):
        X = self._as_2d(X)
        return X @ self.coef_ + self.intercept_




def _prepare(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true и y_pred должны быть одной длины")
    return y_true, y_pred


def mae(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1.0 - ss_res / ss_tot




x = np.array([5, 15, 25, 35, 45, 55]).reshape(-1, 1)
y = np.array([5, 20, 14, 32, 22, 38])

model = ManualLinearRegression().fit(x, y)
y_pred = model.predict(x)

print("intercept_ =", model.intercept_)
print("coef_      =", model.coef_)
print("MAE  =", mae(y, y_pred))
print("MSE  =", mse(y, y_pred))
print("RMSE =", rmse(y, y_pred))
print("R2   =", r2_score(y, y_pred))


from sklearn.linear_model import LinearRegression

sk = LinearRegression().fit(x, y)

assert np.allclose(model.coef_, sk.coef_)
assert np.allclose(model.intercept_, sk.intercept_)
assert np.allclose(model.predict(x), sk.predict(x))