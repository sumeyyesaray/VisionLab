"""VisionLab's single prediction service — not a "mushroom API" or a
"flower API." Every registered model with a 'production' alias in the
MLflow Registry is loaded once at startup into an in-memory cache, keyed by
its registry name; a new domain (e.g. PlantVillage) becomes servable purely
by registering + promoting a model, with zero changes here.

Run with:

    uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000

See scripts/serve.sbatch for running this on TRUBA.
"""

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.predictor import predict
from src.inference.registry import ServableModel, load_all_production_models

MODEL_CACHE: dict[str, ServableModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_CACHE.update(load_all_production_models())
    yield
    MODEL_CACHE.clear()


app = FastAPI(title="VisionLab Model Serving", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": len(MODEL_CACHE)}


@app.get("/models")
def list_models() -> dict:
    return {
        "models": [
            {"name": name, "version": servable.version, "image_size": servable.image_size}
            for name, servable in MODEL_CACHE.items()
        ]
    }


@app.post("/predict")
async def predict_endpoint(
    model_name: str = Query(..., description="Registered model name, e.g. 'mushroom_resnet50'"),
    file: UploadFile = File(...),
) -> dict:
    servable = MODEL_CACHE.get(model_name)
    if servable is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {sorted(MODEL_CACHE)}",
        )

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    return predict(servable, image)
