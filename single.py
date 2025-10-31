import streamlit as st
import pickle
import numpy as np

with open("fuel_encoder.pkl", "rb") as f:
    le_fuel = pickle.load(f)

fuel_input = "Petrol"  # From UI
fuel_encoded = le_fuel.transform([fuel_input])[0]

st.title("Welcome to the car price prediction model")
st.header("Please fill in vehicle details")

vehicle_age       = st.number_input("Vehicle age",0,10,5)
km_driven         = st.number_input("KM driven",0,200)
mileage           = st.number_input("Mileage",0,200)
engine            = st.number_input("Engine size",0,200)
max_power         = st.number_input("Max power",0,200)
seats             = st.number_input("Number of seats",0,200)
seller_type       = st.number_input("Type of seller",0,200)
fuel_type         = st.slider('Type of fuel:', 0, 100, 25)
transmission_type = st.slider('Type ofransmission:', 0, 100, 25)

#seller_type       = st.selectbox("Type of fuel, ["Petrol", "Diesel", "CNG", "LPG", "Electric" ])
#fuel_type         = st.selectbox()
#transmission_type =  st.selectbox()

#create a button to predict output
predict_clicked=st.button("Get the prediction")

if predict_clicked==True:
    model=pickle.load(open("dataset/model_development/RFR.pkl", 'rb'))
    #load the test data into numpy array
    data=[np.array([vehicle_age,km_driven,mileage,engine,max_power,seats, seller_type, fuel_type, transmission_type])]

    #call the model to predict the price
    result=model.predict(data)
    print(result) 
else:
    print("")
    #if (result==1):
    #    result_string = "Diabetic"
    #    st.error("The outcome for your diabetes test is "+result_string)
    #else:
     #   result_string = "Non-Diabetic"
     #   st.success("The outcome for your diabetes test is "+result_string)

    #display the predicted price on the webpage
    
