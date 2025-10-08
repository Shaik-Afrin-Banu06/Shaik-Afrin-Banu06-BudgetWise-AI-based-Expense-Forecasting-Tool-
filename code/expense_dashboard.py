# --------------------------------------------------------
# STREAMLIT DASHBOARD APP
# --------------------------------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 📥 Load data
expenses = pd.read_csv("data/processed_monthly_data.csv")
forecast = pd.read_csv("data/forecast_next_6_months.csv")

st.set_page_config(page_title="BudgetWise Dashboard", layout="wide")

# 🎯 Title
st.title("💰 BudgetWise – AI Expense Forecast Dashboard")

# 🧾 Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("📆 Total Months", len(expenses))
col2.metric("💸 Avg Monthly Expense", f"₹{expenses['total_expenses'].mean():,.0f}")
col3.metric("💰 Avg Savings", f"₹{expenses['net_savings'].mean():,.0f}")

# 📈 Line Chart: Actual vs Forecast
st.subheader("📊 Expense Forecast")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(expenses['year_month'], expenses['total_expenses'], label='Actual Expenses', marker='o')
ax.plot(forecast['year_month'], forecast['predicted_expenses'], label='Forecast', linestyle='--', marker='x')
ax.set_xlabel("Month")
ax.set_ylabel("Expenses (₹)")
ax.legend()
st.pyplot(fig)

# 🗂 Category Selector (Optional Future Feature)
st.info("💡 Future Feature: Category-wise forecasting will be added soon!")

# ✅ Footer
st.caption("Built by Shaik Afrin Banu | AICTE Internship 2025")
