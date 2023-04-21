import streamlit as st
import pandas as pd
import numpy as np
import plost
from PIL import Image

# Page setting
# st.set_page_config(layout="wide")
#
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Data
seattle_weather = pd.read_csv('File_DevUnits_TestUnits.csv')
#stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

# Row A
a1, a2, a3 = st.columns(3)
a1.image(Image.open('D:\IISc\CAPSTONE\N-CMAPSS_DL-main\STREAMLIT images\full bad RUL\N-CMAPSS_DS01-005_unit7_test_w50_s1_bs256_lr0.001_sub10_rmse-14.23.png'))
a2.metric("DS01","Unit", "7")
a3.metric("rmse", "14.23")

# Row B
b1, b2, b3, b4 = st.columns(4)
b1.metric("Temperature", "70 °F", "1.2 °F")
b2.metric("Wind", "9 mph", "-8%")
b3.metric("Humidity", "86%", "4%")
b4.metric("Humidity", "86%", "4%")

# Row C
#c1, c2 = st.columns((7,3))
c1 = st.columns()
with c1:
    st.markdown('N-CMAPSS_DS01-005_unit7_test_w50_s1_bs256_lr0.001_sub10_rmse-14.23.png')
    plost.time_hist(
    data=seattle_weather,
    x_unit='Timestamps',
    y_unit='RUL',
    color='temp_max',
    aggregate='median',
    legend='Predicted, Truth')
# with c2:
#     st.markdown('### Bar chart')
#     plost.donut_chart(
#         data=stocks,
#         theta='q2',
#         color='company')
