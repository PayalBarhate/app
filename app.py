
import streamlit as st 
import joblib
import pandas as pd

model = joblib.load('logistic_regression_model.pkl')

st.title("Titanic Survival Prediction")

Pclass = st.selectbox("Pclass (Passenger Class)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
SibSp = st.number_input("SibSp (Number of Siblings/Spouses aboard)", min_value=0)
Parch = st.number_input("Parch (Number of Parents/Children aboard)", min_value=0)
Fare = st.number_input("Fare", min_value=0.0)

sex_map = {'male': 0, 'female': 1}
user_input = pd.DataFrame([[Pclass, sex_map[Sex], Age, SibSp, Parch, Fare]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

prediction = model.predict(user_input)
st.write("Prediction:", "Survived" if prediction[0] == 1 else "Not Survived")
