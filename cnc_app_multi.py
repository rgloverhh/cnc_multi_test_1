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
model_hc = load_model('heart_xg_model.pkl')
model_ma = load_model('ma_xg_model.pkl')

def predict_pcp(input1, input2, input3, input4):
    sl = model_pcp.predict([[input1, input2, input3, input4]])
    return sl[0]

def predict_cc(input1, input2, input3, input4):
    sl = model_cc.predict([[input1, input2, input3, input4]])
    return sl[0]

def predict_hc(input1, input2, input3, input4):
    sl = model_hc.predict([[input1, input2, input3, input4]])
    return sl[0]

def predict_ma(input1, input2, input3, input4):
    sl = model_ma.predict([[input1, input2, input3, input4]])
    return sl[0]


def main():
    st.title("CNC Service Level Predictor")
    st.text("Please fill in the responses below to predict service level")
    st.caption("Default values are daily average from May 2024")
    st.sidebar.header("Choose your department")
    selected_model = st.sidebar.radio("Select department:", ["Primary Care", "Cancer Care", "Heart Care", "MA CRT Team"])

    if selected_model == "Primary Care":
        st.text("1. Call Volumes")
        calls_offered = st.number_input(label="Enter a call volume between 500 and 3000", min_value=500, max_value=4000, step=10, value=1970)
        st.text("2. Average Handle Time")
        aht_minutes = st.number_input(label="Enter AHT minutes between 4 and 6", min_value=4, max_value=6, step=1, value=5)
        aht_seconds = st.number_input(label="Enter AHT seconds between 0 and 59", min_value=0, max_value=59, step=1, value=30)
        st.text("3. Not Ready Rate")
        not_ready = st.number_input(label="Enter Not Ready Rate between 15 and 35", min_value=15.0, max_value=35.0, step=0.1, value=22.7)
        st.text("4. FTEs Logged In (Use Power BI CNC Call Metrics Staffing as a guide)")
        ftes_logged_in = st.number_input(label="Enter FTEs between 20 and 40", min_value=20.0, max_value=40.0, step=0.5, value=25.0)
        not_ready_con = not_ready/100
        aht = aht_minutes + (aht_seconds/60)
        sl_prediction_temp = predict_pcp(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        st.header("Primary Care Service Level Prediction")
        if sl_prediction <= 0:
            st.subheader("0%")
        elif sl_prediction >= 100:
            st.subheader("100%")
        else:
            st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)")
        st.sidebar.caption("Data timeframes: 10/3/2022-6/14/2024")
        st.sidebar.caption("Current accuracy: 90%")


    elif selected_model == "Cancer Care":
        calls_offered = st.number_input(label="Enter a call volume between 300 and 1400", min_value=300, max_value=1400, step=10, value=890)
        aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5.5 = 5min 30sec -> 0.05 = 3 sec)", min_value=5.0, max_value=8.0, step=0.05, value=6.00)
        not_ready = st.number_input(label="Not Ready Rate (%)", min_value=15.0, max_value=35.0, step=0.1, value=25.0)
        ftes_logged_in = st.number_input(label="Choose the total number of FTEs logged in for the day (use PowerBI CNC Call Metrics Staffing as a guide)", min_value=7.0, max_value=17.0, step=0.5, value=12.0)
        not_ready_con = not_ready/100
        sl_prediction_temp = predict_cc(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        st.header("Cancer Care Service Level Prediction")
        if sl_prediction <= 0:
            st.subheader("0%")
        elif sl_prediction >= 100:
            st.subheader("100%")
        else:
            st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)") 
        st.sidebar.caption("Data timeframes: 6/3/2022-6/19/2024")
        st.sidebar.caption("Current accuracy: 86%")


    elif selected_model == 'Heart Care':
        calls_offered = st.number_input(label="Enter a call volume between 300 and 1400", min_value=300, max_value=1400, step=10, value=800)
        aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5.5 = 5min 30sec -> 0.05 = 3 sec)", min_value=4.0, max_value=7.0, step=0.05, value=5.25)
        not_ready = st.number_input(label="Not Ready Rate (%)", min_value=15.0, max_value=35.0, step=0.1, value=21.5)
        ftes_logged_in = st.number_input(label="Choose the total number of FTEs logged in for the day (use PowerBI CNC Call Metrics Staffing as a guide)", min_value=7.0, max_value=18.0, step=0.5, value=13.5)
        not_ready_con = not_ready/100
        sl_prediction_temp = predict_hc(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        st.header("Heart Care Service Level Prediction")
        if sl_prediction <= 0:
            st.subheader("0%")
        elif sl_prediction >= 100:
            st.subheader("100%")
        else:
            st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)") 
        st.sidebar.caption("Data timeframes: 1/3/2022-6/20/2024")
        st.sidebar.caption("Current accuracy: 92%")


    elif selected_model == 'MA CRT Team':
        calls_offered = st.number_input(label="Enter a call volume between 200 and 1100", min_value=200, max_value=1100, step=10, value=510)
        aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5.5 = 5min 30sec -> 0.05 = 3 sec)", min_value=5.0, max_value=9.0, step=0.05, value=6.45)
        not_ready = st.number_input(label="Not Ready Rate (%)", min_value=20.0, max_value=45.0, step=0.1, value=34.0)
        ftes_logged_in = st.number_input(label="Choose the total number of FTEs logged in for the day (use PowerBI CNC Call Metrics Staffing as a guide)", min_value=4.0, max_value=15.0, step=0.5, value=7.5)
        not_ready_con = not_ready/100
        sl_prediction_temp = predict_ma(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        st.header("MA CRT Team Service Level Prediction")
        if sl_prediction <= 0:
            st.subheader("0%")
        elif sl_prediction >= 100:
            st.subheader("100%")
        else:
            st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)") 
        st.sidebar.caption("Data timeframes: 7/3/2023-6/27/2024")
        st.sidebar.caption("Current accuracy: 86%")

if __name__ == "__main__":
    main()