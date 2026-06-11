from fastapi import APIRouter

from app.models.esquemas import SolicitudPrediccion, RespuestaPrediccion
from app.controllers import prediccion_controller

router = APIRouter(prefix="/predicciones", tags=["Predicciones"])


@router.post(
    "/",
    response_model=RespuestaPrediccion,
    summary="Realizar predicción de cancelación",
    description=(
        "Recibe los datos de un suscriptor y devuelve si el modelo "
        "predice que cancelará su plan o continuará activo."
    ),
)
def realizar_prediccion(solicitud: SolicitudPrediccion) -> RespuestaPrediccion:
    return prediccion_controller.predecir(solicitud)
