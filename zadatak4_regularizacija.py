import numpy as np
import pandas as pd
from poly_regression import PolynomialRegression

def main():
    data = {
        'X': [520, 610, 700, 820, 910, 1030, 1120, 1250, 1390, 1510],
        'Y': [3.1, 3.8, 4.0, 5.1, 5.3, 6.0, 6.2, 5.9, 5.1, 4.1]
    }
    df = pd.DataFrame(data)
    X = df['X'].values
    y = df['Y'].values

    X_new = np.array([600, 900, 1300, 1600])

    print("=" * 75)
    print("ZADATAK 4: ANALIZA L2 REGULARIZACIJE (RIDGE REGRESIJA)")
    print("=" * 75)

    for M in [3, 5, 9]:
        print(f"\n=================== Stepen polinoma M = {M} ===================")
        print(f"{'Alpha':<10} | {'MSE (Train)':<12} | {'Norma ||w||':<14} | {'Predikcije (600, 900, 1300, 1600 mm)'}")
        print("-" * 75)
        for alpha in [0.0, 1e-5, 1e-3, 1.0, 100.0]:
            model = PolynomialRegression(degree=M, alpha=alpha)
            model.fit(X, y)
            
            mse = model.score(X, y)
            preds = model.predict(X_new)
            norm_w = np.linalg.norm(model.coef_)
            
            preds_str = ", ".join([f"{p:.2f}" for p in preds])
            print(f"{alpha:<10} | {mse:<12.6f} | {norm_w:<14.4e} | [{preds_str}]")

if __name__ == "__main__":
    main()
