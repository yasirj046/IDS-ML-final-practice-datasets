import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Medicine Sales Forecast",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Medicine Sales Forecasting Tool")
st.markdown("""
This application allows you to predict next month's sales for a specific medicine based on historical sales data.
Enter the medicine name and past sales figures to get started.
""")

medicine_name = st.text_input("Enter medicine name:", "")

st.subheader("Enter Historical Sales Data")

def generate_date_options():
    options = []
    current_date = datetime.datetime.now()
    for i in range(60):  # 5 years back
        date = current_date - pd.DateOffset(months=i)
        options.append(f"{date.strftime('%b')} {date.year}")
    return options

date_options = generate_date_options()

if 'sales_data' not in st.session_state:
    st.session_state.sales_data = []
    start_date = datetime.datetime.now()
    for i in range(3):
        row_date = start_date + pd.DateOffset(months=i)
        st.session_state.sales_data.append({
            'date_str': f"{row_date.strftime('%b')} {row_date.year}",
            'date': row_date.strftime('%Y-%m-%d'),
            'sales': ""
        })

def update_sales(index, sales_value):
    st.session_state.sales_data[index]['sales'] = sales_value

def update_date(index, new_date_str):
    try:
        date_obj = datetime.datetime.strptime(new_date_str, '%b %Y')
        for i in range(len(st.session_state.sales_data)):
            new_date = date_obj + pd.DateOffset(months=i)
            st.session_state.sales_data[i]['date_str'] = f"{new_date.strftime('%b')} {new_date.year}"
            st.session_state.sales_data[i]['date'] = new_date.strftime('%Y-%m-%d')
    except ValueError:
        st.error(f"Invalid date format: {new_date_str}")

def add_empty_row():
    if st.session_state.sales_data:
        last_dt = datetime.datetime.strptime(st.session_state.sales_data[-1]['date'], "%Y-%m-%d")
        next_dt = last_dt + pd.DateOffset(months=1)
        st.session_state.sales_data.append({
            'date_str': f"{next_dt.strftime('%b')} {next_dt.year}",
            'date': next_dt.strftime('%Y-%m-%d'),
            'sales': ""
        })

def remove_sales_entry(index):
    st.session_state.sales_data.pop(index)

def clear_all_data():
    st.session_state.sales_data = []
    start_date = datetime.datetime.now()
    for i in range(3):
        row_date = start_date + pd.DateOffset(months=i)
        st.session_state.sales_data.append({
            'date_str': f"{row_date.strftime('%b')} {row_date.year}",
            'date': row_date.strftime('%Y-%m-%d'),
            'sales': ""
        })

for i, row in enumerate(st.session_state.sales_data):
    col1, col2, col3 = st.columns([4, 3, 1])
    with col1:
        if i == 0:
            selected_date = st.selectbox(
                f"Month-Year #{i + 1}",
                options=date_options,
                index=date_options.index(row['date_str']) if row['date_str'] in date_options else 0,
                key=f"date_{i}",
            )
            if selected_date != row['date_str']:
                update_date(i, selected_date)
        else:
            st.text_input(f"Month-Year #{i + 1}", value=row['date_str'], key=f"date_{i}", disabled=True)
    with col2:
        sales_value = st.number_input(
            f"Sales #{i + 1}",
            min_value=0.0,
            value=float(row['sales']) if row['sales'] and str(row['sales']).replace('.', '', 1).isdigit() else 0.0,
            key=f"sales_{i}"
        )
        if sales_value != row['sales']:
            update_sales(i, sales_value)
    with col3:
        if len(st.session_state.sales_data) > 3:
            st.button("❌", key=f"remove_{i}", on_click=remove_sales_entry, args=(i,))
        else:
            st.write("")

col1, col2 = st.columns([1, 5])
with col1:
    st.button("+ Add Row", on_click=add_empty_row)
with col2:
    st.button("Clear All Data", on_click=clear_all_data)

valid_data_points = 0
for row in st.session_state.sales_data:
    if row['sales'] and str(row['sales']).replace('.', '', 1).isdigit() and float(row['sales']) > 0:
        valid_data_points += 1

forecast_button = st.button("Generate Forecast")

if forecast_button:
    if valid_data_points < 3:
        st.error("Please enter at least 3 valid sales data points.")
    else:
        valid_data = []
        for row in st.session_state.sales_data:
            if row['sales'] and str(row['sales']).replace('.', '', 1).isdigit() and float(row['sales']) > 0:
                valid_data.append({
                    'date': row['date'],
                    'sales': float(row['sales'])
                })
        df = pd.DataFrame(valid_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        if df.duplicated('date').any():
            st.warning("Duplicate dates found. Aggregating sales for duplicate dates.")
            df = df.groupby('date')['sales'].sum().reset_index()
        df.set_index('date', inplace=True)

        # SARIMA fixed parameters
        p, d, q = 1, 1, 1
        P, D, Q, m = 1, 1, 1, 12

        try:
            model = SARIMAX(
                df['sales'],
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False, maxiter=200)
            last_date = df.index[-1]
            # Use get_forecast, and .predicted_mean for actual prediction
            forecast_obj = model_fit.get_forecast(steps=1)
            forecast_val = forecast_obj.predicted_mean.iloc[0]
            # Make sure not to display last observed sales if forecast is identical for all values
            if np.isclose(forecast_val, df['sales'].iloc[-1], rtol=0, atol=1e-6) and not np.allclose(df['sales'], df['sales'].iloc[-1]):
                # If it's just repeating last value and the whole series isn't constant, try a fallback: increase d for trend
                model = SARIMAX(
                    df['sales'],
                    order=(p, min(d+1,2), q),
                    seasonal_order=(P, D, Q, m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False, maxiter=200)
                forecast_obj = model_fit.get_forecast(steps=1)
                forecast_val = forecast_obj.predicted_mean.iloc[0]
            next_month = (last_date + pd.DateOffset(months=1)).replace(day=1)
            st.subheader("Forecast Result")
            st.markdown(
                f"The predicted sales for **{medicine_name}** in **{next_month.strftime('%B %Y')}** is **{forecast_val:,.0f}** units."
            )
        except Exception as e:
            st.error(f"Error in SARIMA modeling: {str(e)}")
else:
    st.info("Enter your sales data and click 'Generate Forecast' to see prediction.")

st.sidebar.header("About")
st.sidebar.info("""
This application uses the SARIMA model to forecast next month's medicine sales based on your historical data.
- Enter at least 3 months of sales
- The first month you select will automatically set the following rows to the next months
- Only the prediction for the next month is shown
""")
st.markdown("---")
st.markdown("*Sales Forecasting Tool Created By Abdullah And Yasir*🌹")