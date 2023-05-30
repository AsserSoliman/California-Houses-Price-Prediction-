import streamlit as st 
import joblib
import pandas as pd
import numpy as np

# load data
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")

# Load the saved models 
lin_reg = joblib.load('lin_reg.pkl')
tree = joblib.load("new_tree.pkl")
fullpipe = joblib.load("full_pipe.pkl")

# create title
st.title("California Housing Price Prediction App")

# Get inputs
longitude = st.slider('Longitude', float(df['longitude'].min()), float(df['longitude'].max()))
latitude = st.slider('Latitude', float(df['latitude'].min()), float(df['latitude'].max()))
housing_median_age = st.slider('Housing median age', float(df['housing_median_age'].min()), float(df['housing_median_age'].max()))
total_rooms = st.slider('Total rooms', float(df['total_rooms'].min()), float(df['total_rooms'].max()))
total_bedrooms = st.slider('Total_bedrooms', float(df['total_bedrooms'].min()), float(df['total_bedrooms'].max()))
population = st.slider('Population', float(df['population'].min()), float(df['population'].max()))
households = st.slider('Households', float(df['households'].min()), float(df['households'].max()))
median_income = st.slider('Median income', float(df['median_income'].min()), float(df['median_income'].max())) 
ocean_proximity = st.selectbox('ocean_proximity', ('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))

# Create dictionary to turn it to data frame 
data = {'longitude':longitude,'latitude':latitude,'housing_median_age':housing_median_age,'total_rooms':total_rooms,'total_bedrooms':total_bedrooms,'population':population,'households':households,'median_income':median_income,'ocean_proximity':ocean_proximity}       

# To data frame
features = pd.DataFrame(data,index=[0])

# Feature Engineering
features['rooms_per_household'] = features['total_rooms']/features['households']
features['bedrooms_per_room'] = features['total_bedrooms']/features['total_rooms']
features['population_per_household'] = features['population']/features['households']

# Pipeline
features_ready = fullpipe.transform(features)

# Predict using Linear Regression
prediction = lin_reg.predict(features_ready)

# Show output 
st.markdown('''# $ {} '''.format(round(prediction), 2))