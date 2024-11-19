import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
with open('regressor.pkl', 'rb') as file:
    reg = pickle.load(file)

# Load the label encoders
lb = LabelEncoder()
lb.classes_ = np.array(['Diesel', 'Petrol', 'CNG', 'LPG'])  # Adjust based on your data

lb2 = LabelEncoder()
lb2.classes_ = np.array(['Dealer', 'Individual', 'Trustmark Dealer'])  # Adjust based on your data

lb1 = LabelEncoder()
lb1.classes_ = np.array(['Automatic', 'Manual'])  # Adjust based on your data

# Streamlit App Title
st.title("Car Price Prediction App")

# Input fields for user
year = st.number_input("Enter the Year of Manufacture:", min_value=1900, max_value=2024, step=1)
mileage = st.text_input("Enter the Mileage (in km):")
fuel_type = st.selectbox("Select the Fuel Type:", lb.classes_)
seller_type = st.selectbox("Select the Seller Type:", lb2.classes_)
transmission = st.selectbox("Select the Transmission Type:", lb1.classes_)

# Prediction button
if st.button("Predict"):
    try:
        # Prepare the input data
        new_data = [year, mileage, fuel_type, seller_type, transmission]
        new_data[2] = lb.transform([new_data[2]])[0]
        new_data[3] = lb2.transform([new_data[3]])[0]
        new_data[4] = lb1.transform([new_data[4]])[0]

        # Convert to numpy array
        new_data = np.array(new_data, dtype=float).reshape(1, -1)

        # Predict the car price
        result = reg.predict(new_data)
        st.success(f"The Predicted Car Price is: ₹{result[0]:,.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
