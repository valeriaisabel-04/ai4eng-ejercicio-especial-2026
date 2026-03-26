import numpy as np
import random
from sklearn.ensemble import IsolationForest


# =========================================================
# FUNCIÓN GENERADORA
# =========================================================
def generar_caso_de_uso_limpiar_anomalias_entrenamiento():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función limpiar_anomalias_entrenamiento.
    """

    # ---------------------------------------------------------
    # 1. Dimensiones aleatorias
    # ---------------------------------------------------------
    n_samples = random.randint(50, 120)
    n_features = random.randint(2, 6)

    # ---------------------------------------------------------
    # 2. Generar datos
    # ---------------------------------------------------------
    X = np.random.randn(n_samples, n_features)

    # Insertar algunas anomalías fuertes manualmente 👀
    n_outliers = random.randint(1, int(n_samples * 0.2))
    indices_outliers = np.random.choice(n_samples, n_outliers, replace=False)
    X[indices_outliers] += np.random.uniform(8, 15)  # valores extremos

    # Contaminación aleatoria (máx 0.5)
    contaminacion = round(random.uniform(0.05, 0.4), 2)

    # ---------------------------------------------------------
    # 3. INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy(),
        "contaminacion": contaminacion
    }

    # ---------------------------------------------------------
    # 4. OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    model = IsolationForest(
        contamination=contaminacion,
        random_state=42
    )

    pred = model.fit_predict(X)

    # índices de datos normales (1)
    indices_normales = np.where(pred == 1)[0]

    output_data = np.array(indices_normales)

    return input_data, output_data


# =========================================================
# FUNCIÓN SOLUCIÓN
# =========================================================
def limpiar_anomalias_entrenamiento(X, contaminacion):

    model = IsolationForest(
        contamination=contaminacion,
        random_state=42
    )

    pred = model.fit_predict(X)

    # devolver índices de los normales
    return np.where(pred == 1)[0]


# =========================================================
# VALIDACIÓN
# =========================================================
if __name__ == "__main__":

    print("=== VERIFICANDO ALEATORIEDAD ===")

    for i in range(3):
        entrada, salida = generar_caso_de_uso_limpiar_anomalias_entrenamiento()

        print(f"\nCaso {i+1}:")
        print(f"  - Shape X: {entrada['X'].shape}")
        print(f"  - Contaminación: {entrada['contaminacion']}")
        print(f"  - Índices normales (primeros 5): {salida[:5]}")

    print("\n" + "="*50)
    print("VALIDANDO 10 CASOS DE PRUEBA")
    print("="*50)

    exitos = 0

    for i in range(1, 11):
        entrada, esperado = generar_caso_de_uso_limpiar_anomalias_entrenamiento()
        obtenido = limpiar_anomalias_entrenamiento(**entrada)

        if np.array_equal(esperado, obtenido):
            print(f"Caso {i:02d}: OK ✅")
            exitos += 1
        else:
            print(f"Caso {i:02d}: ERROR ❌")

    print("="*50)
    print(f"RESULTADO FINAL: {exitos}/10 casos correctos")
