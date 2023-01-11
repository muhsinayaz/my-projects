import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline



st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra',
             'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))
age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
type = st.sidebar.selectbox("Select usage history", ("Used", "New", "Pre-registered", "Employee's car", "Demonstration"))
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
gears = st.sidebar.selectbox("Select gears", (5,6,7,8))



predict_model=pickle.load(open("lasso_model","rb"))

my_dict = {
    "make_model": car_model,
    "age": age,
    "hp_kW": hp,
    "km": km,
    "type": type,
    'gearing_type':gearing_type,
    "gears": gears
    
}

df = pd.DataFrame.from_dict([my_dict])



st.header("The configuration of your car is below")
st.table(df)


st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = predict_model.predict(df)
    st.success("The estimated price of your car is € {}. ".format(int(prediction[0])))
    # st.success("The estimated price of your car is €{}. ".format(prediction[0]))
