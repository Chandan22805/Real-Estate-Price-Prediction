from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')
society_freq = joblib.load('society_freq.joblib')
feature_names = joblib.load('feature_names.joblib')

AREA_TYPES = ['Plot Area', 'Built-up Area', 'Super built-up Area', 'Carpet Area']
AVAILABILITY_TYPES = ['Ready To Move', 'Under Construction']

class HouseInput(BaseModel):
    total_sqft: float
    bath: int = 0
    balcony: int = 0
    rooms: int = 0
    hall: int = 0
    kitchen: int = 0
    location: str
    society: str = "No Society" 
    area_type: str = "Super built-up Area"
    availability: str = "Ready To Move"
    
@app.post("/predict")
def predict(input_data: HouseInput):
    input_df = pd.DataFrame([input_data.model_dump()])
    
    input_df['sqft_per_room'] = np.divide(
        input_df['total_sqft'],
        input_df['rooms'],
        out=np.zeros_like(input_df['total_sqft']),
        where=(input_df['rooms'] > 0) 
    )
    
    input_df['bed_bath_ratio'] = np.divide(
        input_df['rooms'],
        input_df['bath'],
        out=np.zeros_like(input_df['rooms']),
        where=(input_df['bath'] > 0) 
    )
    
    input_df['location_enc'] = encoder.transform(input_df['location'])
    input_df['society_enc'] = input_df['society'].map(society_freq).fillna(0)
    input_df['loc_soc_int'] = input_df['location_enc'] * input_df['society_enc']
    
    for area in AREA_TYPES:
        input_df[f'area_type_{area}'] = (input_df['area_type'] == area).astype(int)
    for avail in AVAILABILITY_TYPES:
        input_df[f'availability_{avail}'] = (input_df['availability'] == avail).astype(int)
    
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
    input_df = input_df[numeric_cols]

    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    prediction = model.predict(input_df)
    return {"price_per_sqft": float(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)