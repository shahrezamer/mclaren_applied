import streamlit as st


result= [{"Predicted Club":""}]
data_rds = ""
string_data = False
names_from_platform_string = ""

st.markdown("<h1 style='text-align: center; color: green;'>JGA Best Gold Club </h1>", unsafe_allow_html=True)
planet = st.text_input("Planet")
shot_distance = st.text_input("Shot Distance")

if string_data:
    question_area=string_data

if st.button('Send'):
    if not data_rds:
        cv = st.text_area("CV", height=200, value = data_rds, disabled=False)
        similar_job_area = st.text_area("Result", height=200, value = "[]", disabled=False)
        
    # model_object = Model(data_rds, temp)
    # result=model_object.open_ai_model()
    # extracted_similar_job = result[0]["summary_text"].split()

    cv = st.text_area("CV", height=200, value = data_rds, disabled=False)
    
    similar_job_area = st.text_area("Result", height=200, value = result, disabled=False)

