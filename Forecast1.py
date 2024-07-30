import pandas as pd
import pickle
import plotly.express as px
import streamlit as st
import datetime as dt

linear_model = pickle.load(open('linear_model.pkl','rb'))
knn_model = pickle.load(open('knn_model.pkl','rb'))
DT_model = pickle.load(open('DT_model.pkl','rb'))
#XG_model = pickle.load(open('XG_model.pkl','rb'))

st.title("Demand Forecasting")
html_temp="""
<div style = "background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Demand Forecasting</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html= True)
activities=["Linear Regression", "k-NN", "Decision Tree"]
option=st.sidebar.selectbox('Which Model Would you like to use?', activities)
st.subheader(option)

from datetime import date
min_date = date(2018, 1, 1)
max_date = date(2030, 12, 31)
#st.session_state['Starting date'] = min_date

X = st.date_input('Starting date', min_value=min_date, max_value=max_date, key='Starting date')

Y = st.date_input('ending date', min_value=min_date, max_value=max_date, key='ending date')
forecast_data = pd.MultiIndex.from_product([
    pd.date_range(start= X, end= Y, freq='D'),
    [i+1 for i in range(10)],
    [i+1 for i in range(50)]], names=['date', 'store', 'item']).to_frame(index=False)
forecast_data['week_days'] = forecast_data['date'].dt.day_name()
forecast_data['year'] = forecast_data['date'].dt.year
forecast_data['month'] = forecast_data['date'].dt.month
forecast_datan = forecast_data.copy()
a ={"Monday" : 0, 'Tuesday' : 1, 'Wednesday' : 2, 'Thursday' : 3, 'Friday' : 4, 'Saturday' : 5, 'Sunday': 6}
e = {7:1, 6: 3, 5:4, 1:7, 4:11, 9:12, 10:15, 3:16, 8:19, 2:21}
c = {1:1, 12:3, 2:4, 3:8, 10:10, 9:12, 4:13, 11:14, 8:17, 5:18, 6:21, 7:24}
d = {5:1, 1:4, 41:5, 47:6, 4:7, 27:8, 16:11, 34:12, 40:16, 37:17, 23: 18, 49:19 \
     , 44:20, 17:23, 3:26, 42:27, 21:30, 30:31, 19:32, 39:35, 32:36, 20:39, 26:40, 43:43, 48:44, 9:45, 6:49\
     , 7:50, 2:51, 46:52, 31:53, 14:54, 35:59, 50:60, 24:61, 33:63, 29:64, 12:65, 11:66, 10:68, 8:70, 36:71, 22:73, 38:74\
     , 45:75, 25:76, 18:78, 13:79, 28:82, 15:83}
forecast_data['week_days'] = forecast_data['week_days'].map(a)
forecast_data['store'] = forecast_data['store'].map(e)
forecast_data['month'] = forecast_data['month'].map(c)
forecast_data['item'] = forecast_data['item'].map(d)
forecast_data['year'] = (forecast_data['year']-2013)+1
forecast_data = forecast_data[['date', 'week_days', 'year', 'month', 'store', 'item']]


forecast_data1 = forecast_data.copy()
forecast_data1.drop(forecast_data.columns[[0]], axis=1, inplace=True)
Forecast_data_val = forecast_data1.values

def final_pred(y_pred, forecast_datan):
    forecast_data['Predicted'] = y_pred.astype(int)
    Predicted_pred = forecast_data.groupby(forecast_data["date"].dt.date)["Predicted"].sum()
    df_pred1 = pd.DataFrame({'Predicted': Predicted_pred})
    date_pred = df_pred1.index.values.tolist()
    df_pred1['date'] = date_pred
    fig = px.line(df_pred1, x='date', y=["Predicted"])
    fig.update_layout(width=800)
    st.write(fig)
    
    forecast_datan['Predicted'] = y_pred.astype(int)
    ITEMS = forecast_datan['item'].unique().tolist()
    for i in ITEMS:
        st.write("For the item", i)
        st.write('%%%%%%%%%%%%%%%%%%%')
        st.write(forecast_datan[forecast_datan['item']== i].groupby('month').Predicted.sum())
        st.write('------------------------------------------------')

if st.button("Classify"):
    if option == "Linear Regression":
        st.write(final_pred(linear_model.predict(Forecast_data_val),forecast_datan))
    elif option == "k-NN":
        st.write(final_pred(knn_model.predict(Forecast_data_val),forecast_datan))
    
    else:
        st.write(final_pred(DT_model.predict(Forecast_data_val),forecast_datan))        
