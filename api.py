from fastapi import FastAPI, Depends
from pydantic import BaseModel

from horoscope_model import HoroscopeModel, get_model

app = FastAPI()


class HoroscopeRequest(BaseModel):
    horoscope_id: int


class HoroscopeResponse(BaseModel):
    horoscope: str


@app.post("/predict", response_model=HoroscopeResponse)
def predict(request: HoroscopeRequest, horoscope_model: HoroscopeModel = Depends(get_model)):
    horoscope_prediction = horoscope_model.generate_horoscope(request.horoscope_id)
    return HoroscopeResponse(horoscope=horoscope_prediction)
