from enum import Enum
from pydantic import BaseModel, Field


class TipoPlan(str, Enum):
    """Categorías válidas para el tipo de plan de suscripción."""
    basico   = "Basico"
    estandar = "Estandar"
    premium  = "Premium"


class SolicitudPrediccion(BaseModel):
    """Datos de entrada que el cliente envía para obtener una predicción."""
    meses_suscrito          : float = Field(..., ge=0,        description="Meses activo en la plataforma")
    peliculas_vistas_mes    : float = Field(..., ge=0,        description="Películas vistas el último mes")
    series_vistas_mes       : float = Field(..., ge=0,        description="Series vistas el último mes")
    horas_uso_diario        : float = Field(..., ge=0,        description="Promedio de horas de uso por día")
    calificacion_promedio   : float = Field(..., ge=1.0, le=5.0, description="Calificación promedio dada (1–5)")
    dispositivos_vinculados : float = Field(..., ge=1,        description="Dispositivos registrados en la cuenta")
    interrupciones_mes      : float = Field(..., ge=0,        description="Eventos de buffering en el mes")
    quejas_soporte          : float = Field(..., ge=0,        description="Tickets de soporte abiertos")
    precio_plan_mensual     : float = Field(..., gt=0,        description="Precio mensual del plan")
    tipo_plan               : TipoPlan

    model_config = {
        "json_schema_extra": {
            "example": {
                "meses_suscrito"         : 3,
                "peliculas_vistas_mes"   : 1,
                "series_vistas_mes"      : 0,
                "horas_uso_diario"       : 0.4,
                "calificacion_promedio"  : 2.0,
                "dispositivos_vinculados": 1,
                "interrupciones_mes"     : 12,
                "quejas_soporte"         : 4,
                "precio_plan_mensual"    : 160.0,
                "tipo_plan"              : "Premium",
            }
        }
    }


class RespuestaPrediccion(BaseModel):
    """Resultado devuelto por el modelo tras la predicción."""
    cancela                  : bool
    etiqueta                 : str
    probabilidad_cancelacion : float
    probabilidad_continuidad : float


class RespuestaEntrenamiento(BaseModel):
    """Resumen del proceso de entrenamiento y métricas obtenidas."""
    mensaje          : str
    registros_train  : int
    registros_test   : int
    exactitud        : float
    precision        : float
    recall           : float
    f1_score         : float


class RespuestaEstado(BaseModel):
    """Estado general de la API y disponibilidad del modelo."""
    estado         : str
    modelo_cargado : bool
    version        : str
