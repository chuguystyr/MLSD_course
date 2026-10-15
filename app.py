from fastapi import FastAPI, HTTPException
from predict import META, predict

app = FastAPI(title="Прогноз згоди на депозит")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": META["model"],
        "threshold": META["threshold"],
        "created": META["created"],
    }


@app.post("/predict")
def predict_endpoint(records: list[dict]):
    try:
        result = predict(records)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return result.to_dict(orient="records")
