import pandas as pd
import numpy as np

def detectar_outliers_produccion(df, columna, metodo='iqr'):
    """
    Detecta outliers por máquina en una columna de producción.

    Parámetros:
        df (pd.DataFrame): Debe contener 'maquina_id', 'fecha' y la columna indicada.
        columna (str): Nombre de la columna numérica a evaluar.
        metodo (str): 'iqr' o 'zscore'.

    Retorna:
        pd.DataFrame: DataFrame original con columna adicional 'outlier'.
    """

    # Copiar para no modificar el original directamente
    resultado = df.copy()

    def marcar_outliers(series):
        if metodo == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            return (series < lower) | (series > upper)

        elif metodo == 'zscore':
            media = series.mean()
            std = series.std()

            # Igual que el generador
            if std == 0:
                return pd.Series(False, index=series.index)

            z = (series - media) / std
            return np.abs(z) > 3

        else:
            raise ValueError("metodo debe ser 'iqr' o 'zscore'")

    resultado['outlier'] = (
        resultado
        .groupby('maquina_id')[columna]
        .transform(marcar_outliers)
    )

    return resultado