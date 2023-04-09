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

html_8 = """
<div style="background-color:#D5DBDB;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ทำนายข้อมูล</h4></center>
</div>
"""

ag=st.slider("กรุณาเลือกข้อมูล Age")
al_in=st.slider("กรุณาเลือกข้อมูล Annual_Income")
sp_sc=st.number_input("กรุณาเลือกข้อมูล Spending_Score")
wk_exp=st.number_input("กรุณาเลือกข้อมูล Work_Experience")
fa_si=st.number_input("กรุณาเลือกข้อมูล Family_Size")

if st.button("ทำนายผล"):
   
   X = dt.drop('Profession', axis=1)
   y = dt.Profession
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)   

   x_input = np.array([[ag, al_in, sp_sc, wk_exp, fa_si]])
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