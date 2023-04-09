from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.header('*Kumpon* : *Prediction of flowers*')

html_8 = """
<div style="background-color:#D5DBDB;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>การทำนายข้อมูลดอกไม้</h4></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./Data/Customer.csv")
st.write(dt.head(100))

dt1 = dt['Age'].sum()
dt2 = dt['Annual_Income'].sum()
dt3 = dt['Spending_Score'].sum()
dt4 = dt['Work_Experience'].sum()
dt5 = dt['Family_Size'].sum()

dx = [dt1, dt2, dt3, dt4, dt5]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4", "d5"])
if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
   st.write("ไม่แสดงข้อมูล")