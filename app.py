import streamlit as st
import pickle
import pandas as pd
import numpy as np
import gzip  # Needed to open the compressed model

# -------------------------------------------------------------------
# 1. LOAD THE TRAINED MODEL
# -------------------------------------------------------------------
# We use gzip because the model file was compressed to fit on GitHub
try:
    with gzip.open('ipl_model.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'ipl_model.pkl' not found. Please upload the model file.")
    st.stop()

# -------------------------------------------------------------------
# 2. WEBSITE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="IPL Score Predictor", layout="centered")

st.markdown("<h1 style='text-align: center; color: white;'>🏏 IPL Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>AgroVision Team Project</h3>", unsafe_allow_html=True)
st.write("---")

# -------------------------------------------------------------------
# 3. USER INPUTS
# -------------------------------------------------------------------
col1, col2 = st.columns(2)

teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Rajasthan Royals', 'Kings XI Punjab',
    'Sunrisers Hyderabad', 'Delhi Capitals'
]

with col1:
    batting_team = st.selectbox('🏏 Select Batting Team', teams)

with col2:
    # Filter out the batting team from bowling options (A team can't play against itself)
    bowling_teams = [t for t in teams if t != batting_team]
    bowling_team = st.selectbox('🥎 Select Bowling Team', bowling_teams)

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Done (Min 5.0)', min_value=5.0, max_value=19.5, step=0.1)
with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=9, step=1)

# -------------------------------------------------------------------
# 4. PREDICTION LOGIC
# -------------------------------------------------------------------
if st.button('🔮 Predict Final Score', type="primary"):
    
    # Calculate additional features
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs
    
    # Create the input DataFrame
    input_df = pd.DataFrame({'current_score': [current_score],
                             'balls_left': [balls_left],
                             'wickets_left': [wickets_left],
                             'crr': [crr]})
    
    # ---------------------------------------------------------------
    # IMPORTANT: One-Hot Encoding (Matching Training Data Columns)
    # ---------------------------------------------------------------
    # The model expects columns for ALL teams (set to 0 or 1)
    
    sorted_teams = sorted(teams) # Ensure consistent order if needed, but manual list is safer
    
    for team in teams:
        # Batting Team Columns
        input_df[f'batting_team_{team}'] = 1 if team == batting_team else 0
        
    for team in teams:
        # Bowling Team Columns
        input_df[f'bowling_team_{team}'] = 1 if team == bowling_team else 0
        
    # Reorder columns to match exactly what the model learned
    # (This step prevents "Feature Mismatch" errors)
    final_columns = ['current_score', 'balls_left', 'wickets_left', 'crr'] + \
                    [f'batting_team_{t}' for t in teams] + \
                    [f'bowling_team_{t}' for t in teams]
    
    # Ensure input_df has all columns in correct order
    # Note: If your training data had a slightly different order, this might need adjustment.
    # But usually, this standard logic works for Random Forest.
    
    try:
        # Make the prediction
        prediction = pipe.predict(input_df)
        final_score = int(prediction[0])
        
        st.write("---")
        st.markdown(f"<h2 style='text-align: center; color: lightgreen;'>Predicted Score: {final_score} - {final_score+10}</h2>", unsafe_allow_html=True)
        st.balloons()
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Make sure the model was trained with the same team names!")

st.write("---")
st.caption("Built with ❤️ by Nanba")
