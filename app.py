import streamlit as st
from utils import locations
import numpy as np
import pandas as pd
import joblib


model = joblib.load('pipline.joblib') 

st.title('House price in Tehran')
# Area,Room,Parking,Warehouse,Elevator,Address
Area = st.slider("Choose Area",0,500) 
Room = st.selectbox("Choose Number of Rooms", [0,1,2,3,4,5])
Parking = st.selectbox("Does it have Parking?", [0,1])
Warehouse = st.selectbox("Does it have Warehouse?", [0,1])
Elevator = st.selectbox("Does it have Elevator?", [0,1])
Address = st.selectbox("Where is the location?", locations) 

columns = ["Area","Room","Parking","Warehouse","Elevator","Address"]


def predict(): 

    row = np.array([Area,Room,Parking,Warehouse,Elevator, Address]) 

    df = pd.DataFrame([row], columns = columns)
    
    # X= df.drop(columns = 'Address', inplace = True)
    prediction = model.predict(df)
    st.write("The price is: ",prediction) 
if st.button('Predict'):
    predict()


