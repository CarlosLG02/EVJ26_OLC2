from fastapi import FastAPI

from app.views.prediccion_router import router as prediccion_router
from app.views.modelo_router     import router as modelo_router
from app.controllers.prediccion_controller import modelo_disponible
from app.models.esquemas import RespuestaEstado

app = FastAPI(
    title="Streaming Churn API",
    description="Demo de FastAPI con patrón MVC para predicción de cancelación de suscripción.",
    version="1.0.0",
)

app.include_router(modelo_router)
app.include_router(prediccion_router)


@app.get("/", response_model=RespuestaEstado, tags=["Estado"])
def estado() -> RespuestaEstado:
    """Verifica que la API está activa e informa si el modelo está disponible."""
    return RespuestaEstado(
        estado         = "activo",
        modelo_cargado = modelo_disponible(),
        version        = "1.0.0",
    )
