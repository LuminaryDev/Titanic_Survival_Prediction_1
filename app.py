import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('titanic_xgb_model.pkl')

# UI elements for user input
st.title("Titanic Survival Prediction")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0, max_value=500, value=50)
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Embarked", ['S', 'C', 'Q'])

# Feature engineering for the new input
family_size = sibsp + parch
title = "Mr" if sex == 'male' else "Mrs"  # Simple assumption, you can add logic for other titles

# Manually adding encoded features (replace with your full encoding logic)
data = {
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked_S': 1 if embarked == 'S' else 0,  # One-hot encoding for Embarked
    'Embarked_C': 1 if embarked == 'C' else 0,
    'Embarked_Q': 1 if embarked == 'Q' else 0,
    'FamilySize': family_size,
    'Title_Mr': 1 if title == 'Mr' else 0,
    'Title_Mrs': 1 if title == 'Mrs' else 0,
    'Title_Miss': 1 if title == 'Miss' else 0,
    'Title_Rare': 0,  # Assuming no rare titles for this input
    'HasCabin': 0  # Assuming no cabin information (you may change based on input)
}

# Convert the data into a DataFrame
input_df = pd.DataFrame([data])

# Make prediction
prediction = model.predict(input_df)

# Display the result
if prediction == 1:
    st.write("The model predicts that this passenger survived.")
else:
    st.write("The model predicts that this passenger did not survive.")
