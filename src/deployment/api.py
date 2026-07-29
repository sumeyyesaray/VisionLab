"""VisionLab's single prediction service — not a "mushroom API" or a
"flower API." Every registered model with a 'production' alias in the
MLflow Registry is loaded once at startup into an in-memory cache, keyed by
its registry name; a new domain (e.g. PlantVillage) becomes servable purely
by registering + promoting a model, with zero changes here.

Every /predict call is also logged to the flywheel store (see
src/flywheel/store.py) — image + prediction + confidence — so it can later
be confirmed or corrected via POST /feedback and, once enough verified
examples accumulate, folded into a retraining run (see
scripts/flywheel_retrain.sbatch). A prediction is not a label: nothing gets
used for training until a human calls /feedback on it.

Run with:

    uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000

See scripts/serve.sbatch for running this on TRUBA.
"""

import io
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.flywheel.store import get_pending_verification, get_stats, log_prediction, submit_feedback
from src.inference.predictor import predict
from src.inference.registry import ServableModel, load_all_production_models

MODEL_CACHE: dict[str, ServableModel] = {}
FLYWHEEL_INCOMING_ROOT = Path("datasets/flywheel")


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
            {
                "name": name,
                "version": servable.version,
                "dataset_type": servable.dataset_type,
                "image_size": servable.image_size,
            }
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

    result = predict(servable, image)

    incoming_dir = FLYWHEEL_INCOMING_ROOT / servable.dataset_type / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or ".jpg"
    saved_path = incoming_dir / f"{uuid.uuid4().hex}{suffix}"
    saved_path.write_bytes(image_bytes)

    prediction_id = log_prediction(
        dataset_type=servable.dataset_type,
        model_name=servable.name,
        model_version=servable.version,
        image_path=str(saved_path),
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
    )

    return {**result, "prediction_id": prediction_id}


class FeedbackRequest(BaseModel):
    prediction_id: int
    true_label: str


@app.post("/feedback")
def feedback_endpoint(feedback: FeedbackRequest) -> dict:
    """Confirms or corrects a previous prediction with a human-verified
    label. This is the only way a prediction ever becomes trainable data —
    see src/flywheel/export.py."""
    updated = submit_feedback(feedback.prediction_id, feedback.true_label)
    if not updated:
        raise HTTPException(status_code=404, detail=f"No prediction with id {feedback.prediction_id}")
    return {"prediction_id": feedback.prediction_id, "true_label": feedback.true_label, "status": "recorded"}


@app.get("/flywheel/stats")
def flywheel_stats(dataset_type: str | None = Query(default=None)) -> dict:
    return get_stats(dataset_type)


@app.get("/flywheel/pending")
def flywheel_pending(dataset_type: str | None = Query(default=None)) -> dict:
    """What a reviewer should look at — see RETRAINING_POLICY.md decision 1:
    only predictions the model itself wasn't confident about (confidence <
    CONFIDENCE_REVIEW_THRESHOLD) ever show up here."""
    rows = get_pending_verification(dataset_type)
    return {
        "pending": [
            {
                "prediction_id": row["id"],
                "dataset_type": row["dataset_type"],
                "model_name": row["model_name"],
                "predicted_class": row["predicted_class"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    }
