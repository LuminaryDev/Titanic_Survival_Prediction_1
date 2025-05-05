import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("titanic_xgb_model.pkl")

st.title("🚢 Titanic Survival Prediction")

# Collect user inputs
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
Fare = st.slider("Fare", 0.0, 500.0, 32.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert categorical values
sex_map = {'male': 0, 'female': 1}
embarked_map = {'C': 0, 'Q': 1, 'S': 2}

input_df = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [sex_map[Sex]],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked': [embarked_map[Embarked]]
})

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success("✅ Survived!" if prediction == 1 else "❌ Did not survive")
