import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def preparar_datos(df, target_col):
    # 1. Separar características (X) de la variable objetivo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Seleccionar y discretizar latitud y longitud con KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
    coords = X[['latitud', 'longitud']]
    coords_discretizadas = discretizer.fit_transform(coords)
    
    # 3. Reemplazar los valores originales de latitud y longitud en X
    X = X.copy()
    X['latitud'] = coords_discretizadas[:, 0]
    X['longitud'] = coords_discretizadas[:, 1]
    
    # 4. Convertir X y y a arreglos de numpy
    X = X.to_numpy()
    y = y.to_numpy()
    
    # 5. Devolver X procesada y el vector y
    return X, y