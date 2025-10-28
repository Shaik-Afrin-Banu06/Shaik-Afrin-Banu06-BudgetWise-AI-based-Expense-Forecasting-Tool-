import streamlit as st
import pandas as pd, plotly.express as px, plotly.graph_objects as go, locale, os
from backend.init_db import init_db
from backend.auth_utils import verify_user, get_user_id, register_user
from backend.db_utils import add_transaction, get_transactions, import_csv_for_user, get_category_totals, get_monthly_totals
from backend.forecast_utils import linear_forecast_from_monthly_totals
from time import sleep

st.set_page_config(page_title='Budget Forecasting Tool', layout='wide', page_icon='💠')

# Colorful CSS (minimal)
st.markdown("""
<style>
.stButton>button { background: linear-gradient(90deg,#6C5CE7,#2E86DE); color:white; border-radius:10px; padding:8px 12px; }
input[type="text"], input[type="number"] { padding:8px; border-radius:8px; border:1px solid #e6eef6; }
.dataframe tbody tr td { padding:8px; }
.category-pill { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; margin-right:6px; }
.food{ background:#FFEDD5; color:#C2410C } .rent{ background:#E8F7FF; color:#0B5FFF }
.electric{ background:#FFF7ED; color:#D97706 } .recharge{ background:#F0FDF4; color:#15803D }
.ent{ background:#FDF2F8; color:#9F1239 } .travel{ background:#EEF2FF; color:#3730A3 }
.shop{ background:#FFFBEB; color:#92400E } .med{ background:#FEF3C7; color:#92400E } .others{ background:#F3F4F6; color:#374151 }
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="background:linear-gradient(90deg,#6C5CE7,#2E86DE); padding:12px; border-radius:10px; color:white"><h2>💠 Budget Forecasting Tool</h2></div>', unsafe_allow_html=True)
st.write('')

DB_PATH = os.path.join("backend", "budget_tool.db")
# ensure DB exists (creates demo user too)
init_db()

# Force login on page refresh: ensure fresh session_start state
if 'session_started' not in st.session_state:
    st.session_state['session_started'] = True

# --- Authentication ---
if 'user_id' not in st.session_state:
    st.subheader('Login')
    col1, col2 = st.columns([2,1])
    with col1:
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            with st.spinner('Authenticating...'):
                sleep(0.8)
                if verify_user(email, password):
                    st.session_state['user_id'] = get_user_id(email)
                    st.success('Login successful! Redirecting...')
                    sleep(0.5)
                    st.rerun()
                else:
                    st.error('Invalid credentials')
        st.write('---')
        st.write("Don\\'t have an account? Create below")
        new_email = st.text_input('New Email')
        new_password = st.text_input('New Password', type='password')
        if st.button('Sign Up'):
            if register_user(new_email, new_password):
                st.success('Account created — you can now log in.')
            else:
                st.error('Unable to register (email may exist)')
    with col2:
        st.markdown('### Demo account')
        st.write('Email: Afrin123@gmail.com')
        st.write('Password: 123')
    st.stop()

user_id = st.session_state['user_id']

# Logout (clear session only; DB persists)
if st.sidebar.button('Logout'):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.success('Logged out ✅')
    st.experimental_rerun()

# --- Main UI ---
col1, col2 = st.columns([2,1])
with col1:
    st.markdown('<div class="card"><h3>Add Transaction</h3></div>', unsafe_allow_html=True)
    date = st.date_input('Date', value=pd.to_datetime('today'))
    description = st.text_input('Description', placeholder='e.g., Pizza from Zomato')
    amount = st.number_input('Amount (INR)', min_value=0.0, format='%.2f')
    if st.button('Add Transaction'):
        add_transaction(user_id, date.strftime('%Y-%m-%d'), float(amount), description if description else '')
        st.success('Transaction added ✅')
with col2:
    st.markdown('<div class="card"><h3>Import CSV</h3><p class="muted">Upload file with date & amount columns</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded is not None:
        try:
            n = import_csv_for_user(user_id, uploaded)
            st.success(f'Imported {n} rows')
        except Exception as e:
            st.error(str(e))

st.write('')

# Transactions table and metrics
df = get_transactions(user_id)
if df.empty:
    st.info('No transactions yet. Add one!')
else:
    def format_inr(x): return '₹' + f"{x:,.2f}"
    df_display = df.copy()
    df_display['amount'] = df_display['amount'].apply(format_inr)
    st.subheader('Recent Transactions')
    rows_html = ''
    for _, r in df_display.iterrows():
        cat = r['category'] if r['category'] else 'Others'
        cls = 'others'
        if cat == 'Food': cls = 'food'
        elif cat == 'Rent': cls = 'rent'
        elif cat == 'Electricity Bill': cls = 'electric'
        elif cat == 'Recharge Bill': cls = 'recharge'
        elif cat == 'Entertainment': cls = 'ent'
        elif cat == 'Travel': cls = 'travel'
        elif cat == 'Shopping': cls = 'shop'
        elif cat == 'Medical': cls = 'med'

        rows_html += f"<tr><td>{r['date']}</td><td>{r['description']}</td><td style='text-align:right'>{r['amount']}</td><td><span class='category-pill {cls}'>{cat}</span></td></tr>"
    table_html = f"<table style='width:100%; border-collapse:collapse'><thead><tr><th>Date</th><th>Description</th><th style='text-align:right'>Amount</th><th>Category</th></tr></thead><tbody>{rows_html}</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

# Pie chart
st.markdown('---')
st.subheader('Expense Breakdown by Category')
cat_totals = get_category_totals(user_id)
if not cat_totals:
    st.info('No category data yet')
else:
    labels = list(cat_totals.keys()); values = list(cat_totals.values())
    fig = px.pie(names=labels, values=values, hole=0.4)
    fig.update_traces(textinfo='percent+label', hovertemplate='%{label}: ₹%{value:,.2f} (%{percent})')
    st.plotly_chart(fig, use_container_width=True)

# Forecast
st.markdown('---')
st.subheader('Forecast (Linear Regression)')
labels, totals = get_monthly_totals(user_id)
forecast = linear_forecast_from_monthly_totals(labels, totals, months_ahead=6)
if not forecast['months']:
    st.info('Not enough data for forecast')
else:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=forecast['months'], y=forecast['actual'], mode='lines+markers', name='Actual', line=dict(color='#00A3FF')))
    fig2.add_trace(go.Scatter(x=forecast['months'], y=forecast['forecast'], mode='lines+markers', name='Forecast', line=dict(color='#FF0066', dash='dash')))
    fig2.update_layout(xaxis_title='Month', yaxis_title='Amount (INR)')
    st.plotly_chart(fig2, use_container_width=True)

# Download CSV
if not df.empty:
    st.markdown('---')
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Transactions CSV', data=csv, file_name='transactions.csv', mime='text/csv')
