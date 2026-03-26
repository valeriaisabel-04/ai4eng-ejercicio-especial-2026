import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# FUNCIÓN GENERADORA
# =========================================================
def generar_caso_de_uso_seleccionar_y_predecir():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función seleccionar_y_predecir.
    """

    # 1. Dimensiones aleatorias
    n_samples = random.randint(60, 120)
    n_features = random.randint(4, 9)

    # 2. Generar datos
    X = np.random.randn(n_samples, n_features)

    pesos = np.random.randn(n_features)
    y = (X @ pesos > 0).astype(int)

    k_top = random.randint(1, n_features)

    # 3. INPUT
    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "k_top": k_top
    }

    # 4. OUTPUT esperado
    modelo_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_1.fit(X, y)

    importancias = modelo_1.feature_importances_
    indices_top = np.argsort(importancias)[-k_top:]

    X_filtrado = X[:, indices_top]

    modelo_2 = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_2.fit(X_filtrado, y)

    y_pred = modelo_2.predict(X_filtrado)

    output_data = np.array(y_pred)

    return input_data, output_data


# =========================================================
# FUNCIÓN SOLUCIÓN
# =========================================================
def seleccionar_y_predecir(X, y, k_top):

    modelo_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_1.fit(X, y)

    importancias = modelo_1.feature_importances_
    indices_top = np.argsort(importancias)[-k_top:]

    X_filtrado = X[:, indices_top]

    modelo_2 = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_2.fit(X_filtrado, y)

    return modelo_2.predict(X_filtrado)


# =========================================================
# VALIDACIÓN
# =========================================================
if __name__ == "__main__":

    print("=== VERIFICANDO ALEATORIEDAD ===")

    for i in range(3):
        entrada, salida = generar_caso_de_uso_seleccionar_y_predecir()

        print(f"\nCaso {i+1}:")
        print(f"  - Shape de X: {entrada['X'].shape}")
        print(f"  - k_top: {entrada['k_top']}")
        print(f"  - Primeras predicciones: {salida[:5]}")

    print("\n" + "="*50)
    print("VALIDANDO 10 CASOS DE PRUEBA")
    print("="*50)

    exitos = 0

    for i in range(1, 11):
        entrada, esperado = generar_caso_de_uso_seleccionar_y_predecir()
        obtenido = seleccionar_y_predecir(**entrada)

        if np.array_equal(esperado, obtenido):
            print(f"Caso {i:02d}: OK ✅")
            exitos += 1
        else:
            print(f"Caso {i:02d}: ERROR ❌")

    print("="*50)
    print(f"RESULTADO FINAL: {exitos}/10 casos correctos")
