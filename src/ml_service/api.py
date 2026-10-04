from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
from pydantic import BaseModel, ConfigDict, Field
from ml_service.model import Model, load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application."""
    # Load the model when the application starts
    print(" Loading model during application startup...")
    app.state.model = load_model()
    print(" Model loaded:", app.state.model)
    yield
    # Perform any cleanup if necessary when the application shuts down
    app.state.model = None
    
print(" FastAPI application call to create app with lifespan context manager... 1 ")
print(" FastAPI application call to create app with lifespan context manager... 2 ")
app = FastAPI(
    title="ML Prediction Service",
    version="0.2.0",
    lifespan=lifespan,
)
print(" FastAPI application initialized with model:", app.state)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[float] = Field(min_length=2, max_length=2)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    probability: float
    predicted_class: int
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
def create_prediction(
    request: Request,
    payload: PredictionRequest,
) -> PredictionResponse:
    print(" Received prediction request:", request)  
    print(" Received prediction payload:", payload)
    model: Model = request.app.state.model
    result = model.predict(
        payload.features,
        threshold=payload.threshold,
    )

    return PredictionResponse(
        probability=result.probability,
        predicted_class=result.predicted_class,
        model_version="logistic-demo-1",
    )

@app.get("/health")
def health(request: Request) -> dict[str, str]:
    model = request.app.state.model

    if model is None:
        return {"status": "unhealthy"}

    return {
        "status": "healthy",
        "model_version": model.version,
    }