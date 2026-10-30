import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class PolynomialRegression(BaseEstimator, RegressorMixin):
    """
    Klasa za Polinomijalnu Regresiju sa podrškom za L2 (Ridge) regularizaciju.
    Kompatibilna sa scikit-learn API-jem.
    """
    def __init__(self, degree=1, alpha=0.0):
        self.degree = degree
        self.alpha = alpha
        self.coef_ = None

    def _design_matrix(self, X):
        X_arr = np.asarray(X).flatten()
        # Matrix regresije Phi: Phi_ij = x_i^j za j = 0, ..., M
        Phi = np.vander(X_arr, N=self.degree + 1, increasing=True)
        return Phi

    def fit(self, X, y):
        Phi = self._design_matrix(X)
        N, M_plus_1 = Phi.shape
        
        # Phi^T * Phi
        A = Phi.T @ Phi
        
        # Matrix L2 regularizacije
        I = np.eye(M_plus_1)
        I[0, 0] = 0.0  # ne regularizujemo slobodni član (bias w0)
        
        A_reg = A + self.alpha * I
        b = Phi.T @ y
        
        # Rešavanje analitičkog sistema: (Phi^T * Phi + alpha * I) * w = Phi^T * y
        self.coef_ = np.linalg.lstsq(A_reg, b, rcond=None)[0]
        return self

    def predict(self, X):
        Phi = self._design_matrix(X)
        return Phi @ self.coef_

    def score(self, X, y):
        """Vraća Mean Squared Error (MSE)."""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def err_function(self, X, y):
        """Vraća vrednost funkcije cilja/greške Err(w) = 1/(2N) * sum((y_i - y_hat_i)^2)."""
        return 0.5 * self.score(X, y)

if __name__ == "__main__":
    X_sample = np.array([520, 610, 700, 820, 910, 1030, 1120, 1250, 1390, 1510])
    y_sample = np.array([3.1, 3.8, 4.0, 5.1, 5.3, 6.0, 6.2, 5.9, 5.1, 4.1])
    
    model = PolynomialRegression(degree=2)
    model.fit(X_sample, y_sample)
    print("Test pokrenut uspešno!")
    print("Model M=2 MSE:", model.score(X_sample, y_sample))
