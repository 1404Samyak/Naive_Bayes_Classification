import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

@st.cache
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
    return df

df = load_data()
st.write("Dataset Preview:")
st.dataframe(df.head())

@st.cache
def load_scaler_and_model():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("gnb.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_scaler_and_model()

# Sidebar for user input
st.sidebar.header("User Input Features")

def user_input_features():
    total_bill = st.sidebar.slider("Total Bill", min_value=0.0, max_value=50.0, step=0.1)
    tip = st.sidebar.slider("Tip", min_value=0.0, max_value=20.0, step=0.1)
    size = st.sidebar.slider("Size", min_value=1, max_value=6)
    day_fri = st.sidebar.selectbox("Is it Friday?", ["Yes", "No"]) == "Yes"
    day_sat = st.sidebar.selectbox("Is it Saturday?", ["Yes", "No"]) == "Yes"
    day_sun = st.sidebar.selectbox("Is it Sunday?", ["Yes", "No"]) == "Yes"
    # Encoding the day features as 0 or 1
    day_fri = 1 if day_fri else 0
    day_sat = 1 if day_sat else 0
    day_sun = 1 if day_sun else 0
    data = {
        "total_bill": total_bill,
        "tip": tip,
        "size": size,
        "day_Fri": day_fri,
        "day_Sat": day_sat,
        "day_Sun": day_sun
    }
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()
st.subheader("User Input:")
st.write(user_input)

input_scaled = scaler.transform(user_input)

# Predict the smoker status
prediction = model.predict(input_scaled)
if prediction[0] == 0:
    st.subheader("Prediction: Not a Smoker")
else:
    st.subheader("Prediction: Smoker")
