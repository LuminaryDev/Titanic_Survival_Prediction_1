import joblib
import pandas as pd
import streamlit as st

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
embarked = st.selectbox("Embarked", ['S', 'Q'])

# ========== Expected columns ==========
expected_cols = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin',
    'Embarked_Q', 'Embarked_S', 'FamilySize',
    'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare'
]

# ========== Simple preprocessing function ==========
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    family_size = sibsp + parch

    # Title logic
    if sex == 'female':
        title = 'Mrs' if age >= 18 else 'Miss'
    else:
        title = 'Mr'

    # One-hot encode Embarked
    embarked_s = 1 if embarked == 'S' else 0
    embarked_q = 1 if embarked == 'Q' else 0

    # One-hot encode Title
    title_mr = 1 if title == 'Mr' else 0
    title_mrs = 1 if title == 'Mrs' else 0
    title_miss = 1 if title == 'Miss' else 0
    title_rare = 1 if title == 'Rare' else 0  # unlikely in this case

    data = {
        'Pclass': pclass,
        'Sex': 1 if sex == 'male' else 0,  # encode male=1, female=0
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'HasCabin': 0,  # assume no cabin
        'Embarked_Q': embarked_q,
        'Embarked_S': embarked_s,
        'FamilySize': family_size,
        'Title_Miss': title_miss,
        'Title_Mr': title_mr,
        'Title_Mrs': title_mrs,
        'Title_Rare': title_rare
    }

    df = pd.DataFrame([data])

    # Add missing columns if any
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_cols]

# Preprocess input
input_df = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)

# Make prediction
prediction = model.predict(input_df)

# Display the result
if prediction == 1:
    st.write("✅ The model predicts that this passenger **survived**.")
else:
    st.write("❌ The model predicts that this passenger **did not survive**.")
