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
dt2 = dt['Annual_Income ($)'].sum()
dt3 = dt['Spending Score (1-100)'].sum()
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

html_8 = """
<div style="background-color:#D5DBDB;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ทำนายข้อมูล</h4></center>
</div>
"""

pt_len=st.slider("กรุณาเลือกข้อมูล Age")
pt_wd=st.slider("กรุณาเลือกข้อมูล Annual_Income")
sp_len=st.number_input("กรุณาเลือกข้อมูล Spending_Score")
sp_wd=st.number_input("กรุณาเลือกข้อมูล Work_Experience")
sp_wd=st.number_input("กรุณาเลือกข้อมูล Family_Size")

if st.button("ทำนายผล"):
   
   X = dt.drop('variety', axis=1)
   y = dt.variety
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)   

   x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 'Setosa':
    st.image("./Pic/Set1.jpg", use_column_width=True)
   elif out[0] == 'Virginica':
    st.image("./Pic/Vir.jpg", use_column_width=True)
   elif out[0] == 'Versicolor':
    st.image("./Pic/Col.png", use_column_width=True)
   else:       
    st.writ('xxx')    
   #st.button("ไม่แสดงข้อมูล")
else:
   st.write("ไม่แสดงข้อมูล")