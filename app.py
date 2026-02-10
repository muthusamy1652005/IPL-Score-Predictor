%%writefile app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model we just trained
# (Make sure 'ipl_model.pkl' is in the files section or drive)
# If you haven't saved it yet, run the training code again or load from drive
# For now, let's assume the model variable 'regressor' is in memory
# But since this is a separate process, we MUST load from file.

# Let's handle the model loading carefully
try:
    # Try loading from local file
    pipe = pickle.load(open('ipl_model.pkl', 'rb'))
except:
    st.error("Model file not found! Please run the training code and save 'ipl_model.pkl' first.")
    st.stop()

st.title("🏏 IPL Score Predictor (AgroVision Team)")

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', [
        'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Rajasthan Royals', 'Kings XI Punjab',
        'Sunrisers Hyderabad', 'Delhi Capitals'
    ])

with col2:
    bowling_team = st.selectbox('Select Bowling Team', [
        'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Rajasthan Royals', 'Kings XI Punjab',
        'Sunrisers Hyderabad', 'Delhi Capitals'
    ])

current_score = st.number_input('Current Score', min_value=0, step=1)
overs = st.number_input('Overs Done (must be > 5)', min_value=5.0, max_value=19.5, step=0.1)
wickets = st.number_input('Wickets Out', min_value=0, max_value=9, step=1)

if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame({'current_score': [current_score],
                             'balls_left': [balls_left],
                             'wickets_left': [wickets_left],
                             'crr': [crr]})

    # One-Hot Encoding manual fix
    teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
             'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
             'Delhi Capitals', 'Sunrisers Hyderabad']

    for team in teams:
        # Check batting team
        if team == batting_team:
            input_df[f'batting_team_{team}'] = 1
        else:
            input_df[f'batting_team_{team}'] = 0

        # Check bowling team
        if team == bowling_team:
            input_df[f'bowling_team_{team}'] = 1
        else:
            input_df[f'bowling_team_{team}'] = 0
            
    # Ensure column order matches training
    # (This part is tricky in production, but basic logic works here)
    
    try:
        prediction = pipe.predict(input_df)
        st.success(f"🔮 Predicted Final Score: {int(prediction[0])}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")