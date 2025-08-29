import streamlit as st
import joblib
import pandas as pd

# Load trained logistic regression model
model = joblib.load('logistic_regression_model.pkl')

# App title
st.title("Titanic Survival Prediction")
st.markdown("Enter passenger details below to predict survival probability.")

# User inputs
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
SibSp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", min_value=0)
Parch = st.number_input("Number of Parents/Children aboard (Parch)", min_value=0)
Fare = st.number_input("Fare", min_value=0.0, value=32.2)

# Encode 'Sex' as numeric
sex_map = {'male': 0, 'female': 1}
user_input = pd.DataFrame([[Pclass, sex_map[Sex], Age, SibSp, Parch, Fare]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

# Prediction
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display result
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.success("The passenger would have survived ✅")
else:
    st.error("The passenger would not have survived ❌")

st.subheader("Prediction Probability")
st.write(f"Probability of Survival: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Probability of Not Surviving: {prediction_proba[0][0]*100:.2f}%")
