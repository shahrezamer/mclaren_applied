import streamlit as st
from model import clubPredictor

result= [{"Predicted Club":""}]
st.markdown("<h1 style='text-align: center; color: green;'>JGA Best Gold Club </h1>", unsafe_allow_html=True)
planet = st.text_input("Planet")
shot_distance = st.number_input("Shot Distance")

if st.button('Send'):
    model_object = clubPredictor(planet,shot_distance)
    result = model_object.predict()
    cv = st.text_area("Club", height=2, value = result, disabled=False)

