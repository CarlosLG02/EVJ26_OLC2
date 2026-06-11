import numpy as np
import joblib
from pathlib import Path
from fastapi import HTTPException

from app.models.esquemas import SolicitudPrediccion, RespuestaPrediccion

# Directorio donde se esperan los archivos .joblib generados por el notebook.
ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml"


def _cargar_artefactos():
    """Carga el modelo, scaler y lista de features desde disco.

    Lanza HTTP 503 si los archivos no existen, indicando al cliente
    que debe ejecutar el notebook de entrenamiento primero.
    """
    try:
        modelo   = joblib.load(ML_DIR / "modelo_streaming.joblib")
        scaler   = joblib.load(ML_DIR / "scaler_streaming.joblib")
        features = joblib.load(ML_DIR / "features_streaming.joblib")
        return modelo, scaler, features
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modelo no disponible ({e.filename}). "
                "Ejecuta intro_preprocesamiento_avanzado.ipynb y copia los "
                "archivos .joblib a la carpeta ml/."
            ),
        )


def modelo_disponible() -> bool:
    """Verifica si los tres artefactos del modelo existen en disco."""
    archivos = ["modelo_streaming.joblib", "scaler_streaming.joblib", "features_streaming.joblib"]
    return all((ML_DIR / f).exists() for f in archivos)


def predecir(solicitud: SolicitudPrediccion) -> RespuestaPrediccion:
    """Transforma la solicitud, aplica el modelo y devuelve la predicción.

    Convierte el campo categórico tipo_plan a One-Hot Encoding antes de
    escalar y pasar al modelo, replicando el preprocesamiento del notebook.
    """
    modelo, scaler, features = _cargar_artefactos()

    # Conversión manual de tipo_plan a las columnas OHE generadas por get_dummies.
    datos = {
        "meses_suscrito"          : solicitud.meses_suscrito,
        "peliculas_vistas_mes"    : solicitud.peliculas_vistas_mes,
        "series_vistas_mes"       : solicitud.series_vistas_mes,
        "horas_uso_diario"        : solicitud.horas_uso_diario,
        "calificacion_promedio"   : solicitud.calificacion_promedio,
        "dispositivos_vinculados" : solicitud.dispositivos_vinculados,
        "interrupciones_mes"      : solicitud.interrupciones_mes,
        "quejas_soporte"          : solicitud.quejas_soporte,
        "precio_plan_mensual"     : solicitud.precio_plan_mensual,
        "tipo_plan_Estandar"      : int(solicitud.tipo_plan == "Estandar"),
        "tipo_plan_Premium"       : int(solicitud.tipo_plan == "Premium"),
    }

    # El orden de columnas debe coincidir exactamente con el del entrenamiento.
    X        = np.array([[datos[f] for f in features]])
    X_scaled = scaler.transform(X)

    clase = int(modelo.predict(X_scaled)[0])
    proba = modelo.predict_proba(X_scaled)[0]

    return RespuestaPrediccion(
        cancela                  = bool(clase),
        etiqueta                 = "CANCELACIÓN PROBABLE" if clase == 1 else "CONTINUIDAD PROBABLE",
        probabilidad_cancelacion = round(float(proba[1]), 4),
        probabilidad_continuidad = round(float(proba[0]), 4),
    )
