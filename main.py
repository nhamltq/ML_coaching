from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from training import training
import pandas as pd

app = FastAPI()

@app.get("/")
def 

@app.get("/checkheath", description="check heath sever", tags=["Check heath"])
def check_heath():
    return "Alive!"


@app.post("/train_data", description="Train data upload", tags=["Post data"])
def upload_data(
    target_name: str,
    file: UploadFile = File(...),
):
    data = pd.read_csv(file.file)

    training(data, target_name)
    return "training success"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True)
