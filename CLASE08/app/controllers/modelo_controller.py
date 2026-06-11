import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, precision_score, recall_score, f1_score

from app.models.esquemas import RespuestaEntrenamiento

ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml"
SEED   = 42


def _generar_dataset(n: int = 1_500) -> pd.DataFrame:
    """Genera el dataset sintético de suscriptores con problemas de calidad.

    Replica el mismo pipeline del notebook intro_preprocesamiento_avanzado.
    """
    rng = np.random.default_rng(seed=SEED)

    df = pd.DataFrame({
        "meses_suscrito"         : rng.integers(1, 60, size=n).astype(float),
        "peliculas_vistas_mes"   : rng.integers(0, 30, size=n).astype(float),
        "series_vistas_mes"      : rng.integers(0, 15, size=n).astype(float),
        "horas_uso_diario"       : rng.normal(loc=2.5, scale=1.5, size=n).round(1),
        "calificacion_promedio"  : rng.normal(loc=3.5, scale=1.0, size=n).round(1),
        "dispositivos_vinculados": rng.integers(1, 5, size=n).astype(float),
        "interrupciones_mes"     : rng.integers(0, 20, size=n).astype(float),
        "quejas_soporte"         : rng.integers(0, 5,  size=n).astype(float),
        "precio_plan_mensual"    : rng.normal(loc=120, scale=40, size=n).round(2),
        "tipo_plan"              : rng.choice(["Basico", "Estandar", "Premium"], size=n),
    })

    score = (
        (df["horas_uso_diario"]     < 1.0).astype(int) * 2 +
        (df["quejas_soporte"]       >= 3).astype(int)  * 2 +
        (df["calificacion_promedio"] < 2.5).astype(int)    +
        (df["peliculas_vistas_mes"] < 3).astype(int)
    )
    df["cancela"] = ((score + rng.integers(0, 2, size=n)) >= 3).astype(int)

    # Introducción de problemas de calidad
    for col, pct in {"horas_uso_diario": 0.06, "calificacion_promedio": 0.05,
                     "precio_plan_mensual": 0.05, "meses_suscrito": 0.04}.items():
        idx = rng.choice(df.index, size=int(n * pct), replace=False)
        df.loc[idx, col] = np.nan

    df.loc[rng.choice(df.index, size=20, replace=False), "calificacion_promedio"] = \
        rng.uniform(-1, 0.9, size=20).round(1)
    df.loc[rng.choice(df.index, size=15, replace=False), "calificacion_promedio"] = \
        rng.uniform(5.1, 8, size=15).round(1)
    df.loc[rng.choice(df.index, size=12, replace=False), "precio_plan_mensual"]   = \
        rng.uniform(-50, 0, size=12).round(2)

    return pd.concat([df, df.sample(n=20, random_state=SEED)], ignore_index=True)


def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline de limpieza al dataset generado."""
    df = df.copy()

    # Imputación con mediana para columnas numéricas con posibles outliers
    for col in ["precio_plan_mensual", "horas_uso_diario"]:
        df[col] = df[col].fillna(df[col].median())

    # Imputación con media para distribuciones simétricas
    df["calificacion_promedio"] = df["calificacion_promedio"].fillna(
        df["calificacion_promedio"].mean()
    )

    # Imputación con constante: sin registro = primer mes
    df["meses_suscrito"] = df["meses_suscrito"].fillna(1)

    # Restricción de rangos válidos
    df["calificacion_promedio"] = df["calificacion_promedio"].clip(lower=1.0, upper=5.0)

    # Precios negativos → mediana
    mask = df["precio_plan_mensual"] < 0
    df.loc[mask, "precio_plan_mensual"] = np.nan
    df["precio_plan_mensual"] = df["precio_plan_mensual"].fillna(
        df["precio_plan_mensual"].median()
    )

    df = df.drop_duplicates()
    return df


def entrenar() -> RespuestaEntrenamiento:
    """Genera datos, limpia, entrena el modelo y guarda los artefactos en ml/.

    Este es el flujo completo que en el proyecto CreditGuard se dispara
    al presionar el botón 'Entrenar Modelo'.
    """
    df = _generar_dataset()
    df = _limpiar(df)

    # One-Hot Encoding de la variable categórica
    df = pd.get_dummies(df, columns=["tipo_plan"], drop_first=True, dtype=int)

    features = [c for c in df.columns if c != "cancela"]
    X = df[features].values
    y = df["cancela"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    modelo = RandomForestClassifier(
        n_estimators  = 100,
        max_depth     = 8,
        max_leaf_nodes= 40,
        random_state  = SEED,
        n_jobs        = -1,
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Persistencia de los tres artefactos necesarios para predecir
    ML_DIR.mkdir(exist_ok=True)
    joblib.dump(modelo,   ML_DIR / "modelo_streaming.joblib")
    joblib.dump(scaler,   ML_DIR / "scaler_streaming.joblib")
    joblib.dump(features, ML_DIR / "features_streaming.joblib")

    return RespuestaEntrenamiento(
        mensaje         = "Modelo entrenado y guardado correctamente.",
        registros_train = int(X_train.shape[0]),
        registros_test  = int(X_test.shape[0]),
        exactitud       = round(float(accuracy_score (y_test, y_pred)), 4),
        precision       = round(float(precision_score(y_test, y_pred)), 4),
        recall          = round(float(recall_score   (y_test, y_pred)), 4),
        f1_score        = round(float(f1_score       (y_test, y_pred)), 4),
    )
