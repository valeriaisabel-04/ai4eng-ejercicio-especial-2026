import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random

# --- 1. TU FUNCIÓN (La que estamos evaluando) ---
def evaluar_segmentos_regresion(df, target_col, threshold):
    """
    Entrena un modelo y calcula el MSE para dos grupos: 
    los que están por debajo/igual al threshold y los que están por encima.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # División 80/20 con random_state fijo para reproducibilidad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Máscaras booleanas para segmentar el set de prueba
    grupo_bajo = y_test <= threshold
    grupo_alto = y_test > threshold

    # Cálculo de MSE con manejo de segmentos vacíos
    mse_bajo = mean_squared_error(y_test[grupo_bajo], y_pred[grupo_bajo]) if np.sum(grupo_bajo) > 0 else 0.0
    mse_alto = mean_squared_error(y_test[grupo_alto], y_pred[grupo_alto]) if np.sum(grupo_alto) > 0 else 0.0

    return np.array([float(mse_bajo), float(mse_alto)])


# --- 2. FUNCIÓN GENERADORA (El "Profesor" que crea el examen) ---
def generar_caso_de_uso_evaluar_segmentos_regresion():
    n_rows = random.randint(50, 120)
    n_features = random.randint(2, 5)
    data = np.random.randn(n_rows, n_features)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_cols)
    
    coef = np.random.randn(n_features)
    ruido = np.random.randn(n_rows) * random.uniform(0.5, 2.0)
    target_col = 'target'
    df[target_col] = data @ coef + ruido
    
    threshold = float(np.random.uniform(df[target_col].min(), df[target_col].max()))
    
    input_data = {'df': df.copy(), 'target_col': target_col, 'threshold': threshold}
    
    # Lógica de referencia para generar el 'salida_esperada'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    m_bajo = mean_squared_error(y_test[y_test <= threshold], y_pred[y_test <= threshold]) if np.sum(y_test <= threshold) > 0 else 0.0
    m_alto = mean_squared_error(y_test[y_test > threshold], y_pred[y_test > threshold]) if np.sum(y_test > threshold) > 0 else 0.0
    
    return input_data, np.array([float(m_bajo), float(m_alto)])


# --- 3. VALIDACIÓN: EJEMPLOS Y 10 PRUEBAS ---
if __name__ == "__main__":
    print("--- INICIANDO VALIDACIÓN ---")
    
    # EJEMPLO ÚNICO
    print("\n[EJEMPLO 1: Prueba individual]")
    entrada, salida_correcta = generar_caso_de_uso_evaluar_segmentos_regresion()
    mi_resultado = evaluar_segmentos_regresion(**entrada)
    
    print(f"Threshold usado: {entrada['threshold']:.4f}")
    print(f"Esperado: {salida_correcta}")
    print(f"Obtenido: {mi_resultado}")
    print(f"¿Coinciden?: {'✅ SI' if np.allclose(salida_correcta, mi_resultado) else '❌ NO'}")

    # VALIDACIÓN DE 10 PRUEBAS
    print("\n" + "="*50)
    print("VALIDANDO 10 CASOS ALEATORIOS")
    print("="*50)
    
    exitos = 0
    for i in range(1, 11):
        entrada, esperado = generar_caso_de_uso_evaluar_segmentos_regresion()
        resultado = evaluar_segmentos_regresion(**entrada)
        
        if np.allclose(esperado, resultado):
            status = "PASÓ ✅"
            exitos += 1
        else:
            status = "FALLÓ ❌"
            
        print(f"Prueba {i:02d}: {status} | MSE Bajo: {resultado[0]:.4f} | MSE Alto: {resultado[1]:.4f}")

    print("="*50)
    print(f"RESULTADO FINAL: {exitos}/10 exitosas")