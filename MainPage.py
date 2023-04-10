from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

html_8 = """
<div style="background-color:#D5DBDB;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>การทำนายอาชีพ</h4></center>
</div>
"""
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

with col1:
   st.image("./Pic/Hel.png")
with col2:
   st.image("./Pic/Eng.png")
with col3:
   st.image("./Pic/Law.png")
with col4:
   st.image("./Pic/Ent.png")
with col5:
   st.image("./Pic/Art.jpg")
with col6:
   st.image("./Pic/Doc.png")
with col7:
   st.image("./Pic/Hom.jpg")
with col8:
   st.image("./Pic/Mar.jpg")
with col9:
   st.image("./Pic/Exc.jpg")

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./Data/Customer.csv")
st.write(dt.head(1000))

html_8 = """
<div style="background-color:#D5DBDB;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ทำนายข้อมูล</h4></center>
</div>
"""

al_in=st.number_input("กรุณาเลือกข้อมูล Annual Income ($)")
sp_sc=st.slider("กรุณาเลือกข้อมูล Spending Score (1-100)")
wk_exp=st.slider("กรุณาเลือกข้อมูล Work Experience")
fa_si=st.slider("กรุณาเลือกข้อมูล Family Size")

if st.button("ทำนายผล"):
   
   X = dt.drop('Profession', axis=1)
   y = dt.Profession
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)   

   x_input = np.array([[ag, al_in, sp_sc, wk_exp, fa_si]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 'Healthcare':
    st.image("./Pic/Hel.png", use_column_width=True)
   elif out[0] == 'Engineer':
    st.image("./Pic/Eng.png", use_column_width=True)
   elif out[0] == 'Lawyer':
    st.image("./Pic/Law.png", use_column_width=True)
   elif out[0] == 'Entertainment':       
    st.image("./Pic/Ent.png", use_column_width=True)
   elif out[0] == 'Artist':       
    st.image("./Pic/Art.jpg", use_column_width=True) 
   elif out[0] == 'Doctor':       
    st.image("./Pic/Doc.png", use_column_width=True)    
   elif out[0] == 'Homemaker':       
    st.image("./Pic/Hom.jpg", use_column_width=True)
   elif out[0] == 'Marketing':       
    st.image("./Pic/Mar.jpg", use_column_width=True)
   elif out[0] == 'Executive':       
    st.image("./Pic/Exc.jpg", use_column_width=True)
   #st.button("ไม่แสดงข้อมูล")
else:
   st.write("ไม่แสดงข้อมูล")