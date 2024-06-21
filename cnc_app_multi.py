import numpy as np
import streamlit as st
import pickle
from xgboost import XGBRegressor

#inputs - calls offered, AHT, not ready rate, total ftes, fte callouts

def load_model(mdl):
    with open(mdl, 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

model_pcp = load_model('pcp_xg_model.pkl')
model_cc = load_model('cc_xg_model.pkl')

def predict_pcp(input1, input2, input3, input4):
    sl = model_pcp.predict([[input1, input2, input3, input4]])
    return sl[0]

def predict_cc(input1, input2, input3, input4):
    sl = model_cc.predict([[input1, input2, input3, input4]])
    return sl[0]


def main():
    st.title("CNC Service Level Predictor")
    st.text("Please fill in the responses below to predict primary care service level")
    st.caption("Default values are daily average from May 2024")
    st.sidebar.header("Choose your department")

    if st.sidebar.radio("Select department:", ["Primary Care", "Cancer Care"]) == "Primary Care":
        calls_offered = st.number_input(label="Enter a call volume between 500 and 3000", min_value=500, max_value=4000, step=10, value=1970)
        aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5.5 = 5min 30sec -> 0.1 = 6 sec)", min_value=4.0, max_value=7.0, step=0.05, value=5.50)
        not_ready = st.number_input(label="Not Ready Rate (%)", min_value=15.0, max_value=35.0, step=0.1, value=22.7)
        ftes_logged_in = st.number_input(label="Choose the total number of FTEs logged in for the day (use PowerBI CNC Call Metrics Staffing as a guide)", min_value=20.0, max_value=40.0, step=0.5, value=25.0)
        not_ready_con = not_ready/100
        sl_prediction_temp = predict_pcp(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = round((sl_prediction_temp*100),1)
        st.header("Primary Care Service Level Prediction")
        if sl_prediction <= 0:
            st.subheader("0%")
        elif sl_prediction >= 100:
            st.subheader("100%")
        else:
            st.subheader(f"{sl_prediction}%")
        st.caption("Model: eXtreme Gradient Boosting (XGBoost)")
        st.caption("Data timeframes: 10/3/2022-6/14/2024")
        st.caption("Current accuracy: 90%")

    else:
        calls_offered = st.number_input(label="Enter a call volume between 500 and 3000", min_value=300, max_value=1400, step=10, value=890)
        aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5.5 = 5min 30sec -> 0.1 = 6 sec)", min_value=5.0, max_value=8.0, step=0.05, value=6.00)
        not_ready = st.number_input(label="Not Ready Rate (%)", min_value=15.0, max_value=35.0, step=0.1, value=25.0)
        ftes_logged_in = st.number_input(label="Choose the total number of FTEs logged in for the day (use PowerBI CNC Call Metrics Staffing as a guide)", min_value=7.0, max_value=17.0, step=0.5, value=12.0)
        not_ready_con = not_ready/100
        sl_prediction_temp = predict_pcp(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = round((sl_prediction_temp*100),1)
        st.header("Cancer Care Service Level Prediction")
        if sl_prediction <= 0:
            st.subheader("0%")
        elif sl_prediction >= 100:
            st.subheader("100%")
        else:
            st.subheader(f"{sl_prediction}%")
        st.caption("Model: eXtreme Gradient Boosting (XGBoost)") 
        st.caption("Data timeframes: 6/3/2022-6/19/2024")
        st.caption("Current accuracy: 86%")
