from sys import orig_argv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import json

app = FastAPI()
origins = ["*"]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

class Diabetes(BaseModel):
    Pregnancies : int
    Glucose :int
    BloodPressure: int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int


class Heart(BaseModel):
    Age :int
    Sex:int 
    Cp :int
    Trestbps :int
    Chol:int
    Fbs :int
    restecg:int
    thalach : int
    exang: int
    oldpeak : int
    slope : int
    ca :int 
    thal : int

#class Parkinson(BaseModel):


diabetes_model = pickle.load(open('diabetes_model.sav','rb'))
heart_model  = pickle.load(open('heart_disease_model.sav','rb'))
#parkinson_model = pickle.load(open("parkinsons_model.sav",'rb'))

@app.post('/diabetes_prediction')
def diabetes_pred(input_param :Diabetes):
    input_data = input_param.json()
    input_dict = json.loads(input_data)

    val = [input_dict[i] for i in input_dict]
    prediction = diabetes_model.predict([val])

    if prediction[0]==1:
        return 'The person is diabetic'
    else:
        return 'The person is not diabetic'

@app.post('/heart_disease_prediction')
def heart_pred(input_param :Heart):
    input_data = input_param.json()
    input_dict = json.loads(input_data)

    val = [input_dict[i] for i in input_dict]
    prediction = heart_model.predict([val])

    if prediction[0]==1:
        return 'The person does have Heart Disease'
    else:
        return 'The person does not have Heart Disease'