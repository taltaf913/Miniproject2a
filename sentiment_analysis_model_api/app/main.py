import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app import __version__, schemas

from sentiment_analysis_model.predict import make_prediction
from sentiment_analysis_model import __version__ as model_version


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

filename = None

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {'request': request,})


@app.post("/predict/")
async def predict_input_text(request: Request, q: str):
    
    print(" predict_input_text: "+ q)
    
    # TODO: validatins..
    data_in = q
    
    predicted_sentiment, sentiment_probability = make_prediction(p_input_text = data_in)    
    return templates.TemplateResponse("predict.html", {"request": request,
                                                       "predicted_sentiment": predicted_sentiment,
                                                       "sentiment_probability": sentiment_probability})

@app.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
