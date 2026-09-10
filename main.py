from fastapi import FastAPI
import joblib
from pydantic import BaseModel
model=joblib.load("salary_prediction.pkl")

app=FastAPI()
class SalaryInput(BaseModel):
    Experience:int
    projects:int
    certification:int

@app.post("/predict")
def predict_salary(data:SalaryInput):
    prediction = model.predict([[data.Experience,
                                 data.projects,
                                 data.certification]])
    return {"predict_salary":prediction[0]}