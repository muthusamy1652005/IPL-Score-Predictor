import streamlit as st
import pickle
import pandas as pd
import numpy as np
import gzip

# 1. Load Model
try:
    with gzip.open('ipl_model.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found! Please upload 'ipl_model.pkl' to GitHub.")
    st.stop()

# 2. Page Config
st.set_page_config(page_title="IPL Score Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>🏏 IPL Score Predictor</h1>", unsafe_allow_html=True)

# 3. Inputs
col1, col2 = st.columns(2)

teams = sorted([
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Rajasthan Royals', 'Kings XI Punjab',
    'Sunrisers Hyderabad', 'Delhi Capitals'
])

with col1:
    batting_team = st.selectbox('Select Batting Team', teams)
with col2:
    bowling_team = st.selectbox('Select Bowling Team', teams)

col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Done (Min 5)', min_value=5.0, max_value=19.5, step=0.1)
with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=9, step=1)

# 4. PREDICTION LOGIC (The Fix is Here!)
if st.button('Predict Score'):
    # Basic calculations
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    # Create a basic dictionary with numerical inputs
    input_data = {
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr]
    }
    
    # Create initial DataFrame
    input_df = pd.DataFrame(input_data)

    # --- MAGIC FIX: AUTO-ALIGNMENT ---
    # We ask the model: "What columns do you need?"
    try:
        model_columns = pipe.feature_names_in_
    except AttributeError:
        st.error("Model is too old or wasn't saved correctly. Please re-train.")
        st.stop()

    # We manually create the One-Hot columns based on user selection
    # Then we align everything to match the model's expectation
    for col in model_columns:
        if col not in input_df.columns:
            # If the model wants 'batting_team_CSK', we check if user selected CSK
            if col == f'batting_team_{batting_team}' or col == f'bowling_team_{bowling_team}':
                input_df[col] = 1
            else:
                input_df[col] = 0

    # FORCE the order to be exactly what the model wants
    input_df = input_df[model_columns]
    
    # Predict!
    try:
        prediction = pipe.predict(input_df)
        final_score = int(prediction[0])
        st.success(f"🔮 Predicted Final Score: {final_score}")
        st.balloons()
    except Exception as e:
        st.error(f"Error: {e}")



