
import streamlit as st 
import joblib
import pandas as pd
import numpy as np


# Load the saved models 

lin_reg = joblib.load('lin_reg.pkl')
tree = joblib.load("new_tree.pkl")
fullpipe = joblib.load("full_pipe.pkl")

# create title
