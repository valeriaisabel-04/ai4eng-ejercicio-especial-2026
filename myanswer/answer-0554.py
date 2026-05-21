import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def regresion_con_interaccion(df, target_col, col_ordinal, categorias_orden):
    df = df.copy()
    
    # 1. Codificar la variable ordinal respetando el orden natural
    encoder = OrdinalEncoder(categories=[categorias_orden])
    df[col_ordinal] = encoder.fit_transform(df[[col_ordinal]])
    
    # 2. Crear columna de interacción: col_ordinal * primera columna numérica restante
    # Primera columna numérica distinta a col_ordinal y target_col
    col_numerica = next(
        col for col in df.columns
        if col != col_ordinal and col != target_col and pd.api.types.is_numeric_dtype(df[col])
    )
    df['interaccion'] = df[col_ordinal].to_numpy() * df[col_numerica].to_numpy()
    
    # 3. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 4. Dividir en train/test (75%/25%, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    
    # 5. Entrenar LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 6. Calcular métricas
    r2 = round(model.score(X_test, y_test), 4)
    
    # Coeficiente de la columna interaccion
    idx_interaccion = list(X.columns).index('interaccion')
    coef_interaccion = float(model.coef_[idx_interaccion])
    
    return {
        "r2": r2,
        "coef_interaccion": coef_interaccion
    }