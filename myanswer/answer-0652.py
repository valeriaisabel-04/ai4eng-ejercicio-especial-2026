import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detectar_anomalias_isolation_forest(X, contamination=0.1):
    """
    Detecta anomalías usando Isolation Forest.

    Parámetros:
        X : array-like
            Matriz de características.
        contamination : float, default=0.1
            Proporción esperada de anomalías.

    Retorna:
        tuple:
            - etiquetas (np.ndarray): 1 para normal, -1 para anomalía
            - num_anomalias (int): cantidad total de anomalías detectadas
    """

    # 1. Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Crear y entrenar el modelo
    modelo = IsolationForest(contamination=contamination)
    modelo.fit(X_scaled)

    # 3. Predecir anomalías
    etiquetas = modelo.predict(X_scaled)

    # 4. Contar cuántas anomalías hay
    num_anomalias = np.sum(etiquetas == -1)

    # 5. Retornar resultado
    return etiquetas, num_anomalias