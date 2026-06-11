from fastapi import APIRouter

from app.models.esquemas import RespuestaEntrenamiento
from app.controllers import modelo_controller

router = APIRouter(prefix="/modelo", tags=["Modelo"])


@router.post(
    "/entrenar",
    response_model=RespuestaEntrenamiento,
    summary="Entrenar el modelo",
    description=(
        "Genera el dataset sintético, aplica el pipeline de limpieza, "
        "entrena el Random Forest y guarda los artefactos en disco. "
        "Debe ejecutarse antes de usar el endpoint de predicciones."
    ),
)
def entrenar_modelo() -> RespuestaEntrenamiento:
    return modelo_controller.entrenar()
