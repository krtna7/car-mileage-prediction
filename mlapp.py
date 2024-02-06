#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn .linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


# In[2]:


df= pd.read_csv(r"C:\Users\hp\OneDrive\Documents\Auto MPG Reg.csv")


# In[3]:


df


# In[4]:


df.horsepower= pd.to_numeric(df.horsepower,errors="coerce")


# In[5]:


df.horsepower=df.horsepower.fillna(df.horsepower.median())


# In[6]:


y=df.mpg
X=df.drop(['carname','mpg'],axis=1)


# In[7]:


#Define Multiple models as a Dictionary

models= {"Linear Regression": LinearRegression(),"Decision Tree":DecisionTreeRegressor(),"Random Forest": RandomForestRegressor(),"Gradient Boosting": GradientBoostingRegressor()}


# In[8]:


select_model= st.sidebar.selectbox("Select a ML Model",list(models.keys()))


# In[9]:


if select_model=="Linear Regression":
    model=LinearRegression()
elif select_model=="Decision Tree":
    max_depth= st.sidebar.slider("max_depth",8,6,12)
    model= DecisionTreeRegressor(max_depth=max_depth)
elif select_model=="Random Forest":
    n_extimators= st.sidebar.slider("Number of trees",1,500,50)
    model= RandomForestRegressor(n_estimators=n_extimators)
elif select_model=="Gradient Boosting":
    n_extimators= st.sidebar.slider("Number of trees",1,500,50)
    model= GradientBoostingRegressor(n_estimators=n_extimators)


# In[10]:


model.fit(X,y)


# In[11]:


st.title("Predict Mileage per Gallon")
st.markdown("Model to predict Mileage of a Car")
st.header("Car Features")

col1,col2,col3,col4=st.columns(4)
with col1:
    cylinders= st.slider("cylinders",2,8,1)
    displacement= st.slider("displacement",50,500,10)
with col2:
    horsepower= st.slider("horsepower",50,500,10)
    weight=st.slider("weight",1500,6000,250)
with col3:
    acceleration=st.slider("acceleration",8,25,1)
    modelyear=st.slider("modelyear",70,85,1)
with col4:
    origin=st.slider("origin",1,3,1)


# In[12]:


rsquare= model.score(X,y)
y_pred= model.predict(np.array([[cylinders,displacement,horsepower,weight,acceleration,modelyear,origin]]))


# In[13]:


st.header("ML Model Results")
st.write(f"Selected Model: {select_model}")
st.write(f"Rsquare:{rsquare}")
st.write(f"Prediction:{y_pred}")


# In[ ]:




