# Cancelación de Suscripción

Explicación mas detallada en clase07

## Requisitos previos

- Python 3.10 o superior

---

## Instalación

### 1. Crear y activar el entorno virtual

```bash
# Desde la carpeta 
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar la API

```bash
uvicorn app.main:app --reload
```

La API estará disponible en: `http://127.0.0.1:8000`

---

## Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado de la API y disponibilidad del modelo |
| `POST` | `/modelo/entrenar` | Entrena el modelo y guarda los artefactos en `ml/` |
| `POST` | `/predicciones/` | Predicción individual de cancelación |

### Documentación interactiva

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## Ejemplos de uso

### 1. Verificar estado

```bash
curl http://127.0.0.1:8000/
```

```json
{
  "estado": "activo",
  "modelo_cargado": false,
  "version": "1.0.0"
}
```

### 2. Entrenar el modelo

```bash
curl -X POST http://127.0.0.1:8000/modelo/entrenar
```

```json
{
  "mensaje": "Modelo entrenado y guardado correctamente.",
  "registros_train": 1180,
  "registros_test": 296,
  "exactitud": 0.8952,
  "precision": 0.8814,
  "recall": 0.8733,
  "f1_score": 0.8773
}
```

### 3. Realizar una predicción

```bash
curl -X POST http://127.0.0.1:8000/predicciones/ \
  -H "Content-Type: application/json" \
  -d '{
    "meses_suscrito": 3,
    "peliculas_vistas_mes": 1,
    "series_vistas_mes": 0,
    "horas_uso_diario": 0.4,
    "calificacion_promedio": 2.0,
    "dispositivos_vinculados": 1,
    "interrupciones_mes": 12,
    "quejas_soporte": 4,
    "precio_plan_mensual": 160.0,
    "tipo_plan": "Premium"
  }'
```

```json
{
  "cancela": true,
  "etiqueta": "CANCELACIÓN PROBABLE",
  "probabilidad_cancelacion": 0.87,
  "probabilidad_continuidad": 0.13
}
```
