import streamlit as st
import joblib

model = joblib.load("used_car_random_forest.pkl")


st.title("Used Car Price Prediction")
st.subheader("Random Forest")
car_age = st.number_input(
    "Car Age",
    min_value=0,
    step=1
)

kilometers_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1000
)


if st.button("Predict Price Category"):

    new_car = [[car_age, kilometers_driven]]

    prediction = model.predict(new_car)

    st.success("Prediction: " + prediction[0])
