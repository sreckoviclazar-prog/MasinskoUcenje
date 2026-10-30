import numpy as np
import pandas as pd
from poly_regression import PolynomialRegression

def main():
    # Učitavanje skupa podataka iz Tabele 1
    data = {
        'X': [520, 610, 700, 820, 910, 1030, 1120, 1250, 1390, 1510],
        'Y': [3.1, 3.8, 4.0, 5.1, 5.3, 6.0, 6.2, 5.9, 5.1, 4.1]
    }
    df = pd.DataFrame(data)
    X = df['X'].values
    y = df['Y'].values

    # Nove tačke za analizu predikcija i ekstrapolacije
    X_new = np.array([600, 900, 1300, 1600])

    print("=" * 75)
    print("ZADATAK 3: EVALUACIJA MODELA I IZBOR STEPENA POLINOMA (M)")
    print("=" * 75)

    degrees = [1, 2, 3, 5, 9]

    for M in degrees:
        model = PolynomialRegression(degree=M, alpha=0.0)
        model.fit(X, y)
        
        mse = model.score(X, y)
        err = model.err_function(X, y)
        preds = model.predict(X_new)
        
        print(f"\n>>> STEPEN POLINOMA M = {M} <<<")
        print("Vektor procenjenih parametara w:")
        for j, w_j in enumerate(model.coef_):
            print(f"  w_{j} = {w_j:14.6e}")
        print(f"Funkcija greške Err(w) : {err:.6f}")
        print(f"MSE na skupu za obuku   : {mse:.6f}")
        print("Predikcije za nove tačke:")
        for x_val, y_hat in zip(X_new, preds):
            print(f"  X = {x_val:4d} mm  -->  Y_hat = {y_hat:8.2f} t/ha")

if __name__ == "__main__":
    main()
