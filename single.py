import streamlit as st
import pickle
import numpy as np

st.title("Welcome to the car price prediction model")
st.header("Please fill in vehicle details")

vehicle_age=st.number_input("Enter vehicle age",0,10,5)
km_driven=st.number_input("Enter km driven",0,200)
mileage=st.number_input("Enter mileage",0,200)
engine=st.number_input("Enter engine size",0,200)
max_power = st.number_input("Enter max power",0,200)
seats = st.number_input("Enter number of seats",0,200)
seller_type = st.number_input("Enter the type of seller",0,200)
fuel_type = st.slider('Select your the type of fuel:', 0, 100, 25)
transmission_type = st.slider('Enter the transmission type:', 0, 100, 25)

#create a button to predict output
predict_clicked=st.button("Get the prediction")

if predict_clicked==True:
    model=pickle.load(open("CAR PRICE DATASET/Model Development/RFR.pkl", 'rb'))
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
    
