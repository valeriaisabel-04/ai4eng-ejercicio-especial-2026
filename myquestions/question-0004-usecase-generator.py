import numpy as np
import pandas as pd
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


# =========================================================
# FUNCIÓN GENERADORA
# =========================================================
def generar_caso_de_uso_optimizar_y_proyectar():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función optimizar_y_proyectar.
    """

    # ---------------------------------------------------------
    # 1. Dimensiones aleatorias
    # ---------------------------------------------------------
    n_rows = random.randint(50, 120)
    n_features = random.randint(4, 8)

    # ---------------------------------------------------------
    # 2. Generar datos
    # ---------------------------------------------------------
    data = np.random.randn(n_rows, n_features)
    columns = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame(data, columns=columns)

    # Agregar valores nulos aleatorios
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df[mask] = np.nan

    # Crear target (no se usa en PCA pero se debe separar)
    target_col = "target"
    df[target_col] = np.random.randn(n_rows)

    # n_components válido
    n_components = random.randint(1, n_features)

    # ---------------------------------------------------------
    # 3. INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "target_col": target_col,
        "n_components": n_components
    }

    # ---------------------------------------------------------
    # 4. OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    # A. Separar
    X = df.drop(columns=[target_col])

    # B. Imputar (mediana)
    imputer = SimpleImputer(strategy="median")
    X_imputado = imputer.fit_transform(X)

    # C. Escalar (robusto)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputado)

    # D. PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # E. Varianza explicada acumulada
    varianza = float(np.sum(pca.explained_variance_ratio_))

    output_data = varianza

    return input_data, output_data


# =========================================================
# FUNCIÓN SOLUCIÓN
# =========================================================
def optimizar_y_proyectar(df, target_col, n_components):

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    import numpy as np

    # 1. Separar
    X = df.drop(columns=[target_col])

    # 2. Imputar
    imputer = SimpleImputer(strategy="median")
    X_imputado = imputer.fit_transform(X)

    # 3. Escalar
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputado)

    # 4. PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # 5. Retornar varianza acumulada
    return float(np.sum(pca.explained_variance_ratio_))


# =========================================================
# VALIDACIÓN
# =========================================================
if __name__ == "__main__":

    print("=== VERIFICANDO ALEATORIEDAD ===")

    for i in range(3):
        entrada, salida = generar_caso_de_uso_optimizar_y_proyectar()

        print(f"\nCaso {i+1}:")
        print(f"  - Shape df: {entrada['df'].shape}")
        print(f"  - n_components: {entrada['n_components']}")
        print(f"  - Varianza explicada: {salida:.4f}")

    print("\n" + "="*50)
    print("VALIDANDO 10 CASOS DE PRUEBA")
    print("="*50)

    exitos = 0

    for i in range(1, 11):
        entrada, esperado = generar_caso_de_uso_optimizar_y_proyectar()
        obtenido = optimizar_y_proyectar(**entrada)

        if np.isclose(esperado, obtenido, atol=1e-6):
            print(f"Caso {i:02d}: OK ✅")
            exitos += 1
        else:
            print(f"Caso {i:02d}: ERROR ❌")

    print("="*50)
    print(f"RESULTADO FINAL: {exitos}/10 casos correctos")
