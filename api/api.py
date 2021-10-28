from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from horoscope_model import HoroscopeModel, get_model

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HoroscopeRequest(BaseModel):
    horoscope_id: int


class HoroscopeResponse(BaseModel):
    horoscope: str


@app.post("/predict", response_model=HoroscopeResponse)
def predict(request: HoroscopeRequest, horoscope_model: HoroscopeModel = Depends(get_model)):
    horoscope_prediction = horoscope_model.generate_horoscope(request.horoscope_id)
    return HoroscopeResponse(horoscope=horoscope_prediction)
