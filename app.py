# Importing library
import numpy as np
import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI
from model_features.features import FeatureInputs

# Create the app object
app = FastAPI()

# Load model
pickle_in = open("notebook/diabetes_model.pkl", "rb")
diabetes_model = pickle.load(pickle_in)

# Expose the prediction functionality, make a prediction from the passed
# JSON data and return the predicted feature inputs with the confidence

@app.post('/diabetes_predict')
def predict_diabetes(data:FeatureInputs):
    data = data.dict()
    pregnancies = data['Pregnancies']
    glucose = data['Glucose']
    bloodPressure = data['BloodPressure']
    skinThickness = data['SkinThickness']
    insulin = data['Insulin']
    bmi = data['BMI']
    diabetesPedigreeFunction = data['DiabetesPedigreeFunction']
    age = data['Age']

    input_feature_list = [pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,diabetesPedigreeFunction,age]

    prediction = diabetes_model.predict([input_feature_list])

    if (prediction[0] == 0):
        prediction = "The person is not diabetic"
    else:
        prediction = 'The person is diabetic'
    
    return {"prediction": prediction}

# Run the API with uvicorn
# Will run on http://127.0.0.1:8000/docs

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5000)