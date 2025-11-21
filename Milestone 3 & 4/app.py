# app.py
"""
BudgetWise Pro - Single-file Streamlit app
Features:
- SQLite DB auto-creation (users, categories, transactions, goals)
- 12 auto-categories (keyword rules + TF-IDF similarity fallback)
- Manual add, smart text parse, CSV import (auto-categorize)
- Dashboard: metrics, category pie, monthly bar
- Forecast: prophet monthly forecast (if installed) or linear fallback
- Goals module
- Admin: manage categories & users
- Styled modern teal/blue UI
Author: Shaik Afrin Banu
"""
import os
import re
import sqlite3
import warnings
from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)

# Try importing prophet (optional)
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ========== Configuration & Styles ==========
st.set_page_config(page_title="BudgetWise Pro", layout="wide", page_icon="💠")
# Small CSS to style cards & header
st.markdown(
    """
<style>
body {background: #F2F6FF;}
.card {background: #ffffff; border-radius:12px; padding:14px; box-shadow: 0 3px 10px rgba(2,6,23,0.06);}
.header {background: linear-gradient(90deg,#4B9FFF,#2EC4B6); padding:12px; border-radius:10px; color:white}
.kpi {font-size:20px; font-weight:700}
.small-muted {color:#6b7280; font-size:12px}
.btn {background: linear-gradient(90deg,#4B9FFF,#2EC4B6); color:white; border-radius:8px; padding:6px 10px;}
</style>
""",
    unsafe_allow_html=True,
)

# ========== Database helpers ==========
DB_PATH = "budget_tool.db"


def init_db():
    """Create DB and all tables if missing; insert default categories & demo user."""
    create = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    # Create users
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT
        )
    """
    )
    # Create categories
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """
    )
    # Create transactions
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            amount REAL,
            description TEXT,
            category TEXT,
            type TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """
    )
    # Create goals
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            target_amount REAL,
            current_amount REAL DEFAULT 0,
            target_date TEXT
        )
    """
    )
    conn.commit()

    # Insert default categories if table is empty
    c.execute("SELECT COUNT(*) FROM categories")
    count = c.fetchone()[0]
    if count == 0:
        default_categories = [
            "Food & Dining",
            "Groceries",
            "Travel & Transport",
            "Shopping",
            "Bills & Utilities",
            "Entertainment",
            "Health & Medical",
            "Education",
            "Personal Care",
            "Subscriptions",
            "Savings & Investments",
            "Other",
        ]
        for cat in default_categories:
            try:
                c.execute("INSERT INTO categories (name) VALUES (?)", (cat,))
            except Exception:
                pass
        conn.commit()

    # Insert demo user
    try:
        c.execute(
            "INSERT OR IGNORE INTO users (email, password) VALUES (?, ?)",
            ("Afrin123@gmail.com", "123"),
        )
        conn.commit()
    except Exception:
        pass

    conn.close()


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


init_db()

# ========== Auto-categorization: keywords + TF-IDF fallback ==========
AUTO_CATEGORIES = {
    "Food & Dining": [
        "restaurant",
        "food",
        "pizza",
        "burger",
        "dinner",
        "lunch",
        "coffee",
        "zomato",
        "swiggy",
    ],
    "Groceries": ["grocery", "supermarket", "vegetable", "bread", "milk", "bigbasket"],
    "Travel & Transport": [
        "uber",
        "ola",
        "cab",
        "taxi",
        "bus",
        "train",
        "flight",
        "ticket",
        "fuel",
        "petrol",
    ],
    "Shopping": ["amazon", "flipkart", "myntra", "shopping", "clothes", "shirt"],
    "Bills & Utilities": [
        "electricity",
        "water bill",
        "gas bill",
        "broadband",
        "internet",
        "wifi",
        "rent",
    ],
    "Entertainment": ["movie", "netflix", "prime", "ott", "concert", "game"],
    "Health & Medical": ["doctor", "hospital", "pharmacy", "medicine"],
    "Education": ["course", "udemy", "coursera", "tuition", "college"],
    "Personal Care": ["salon", "spa", "haircut", "barber", "beauty"],
    "Subscriptions": ["subscription", "spotify", "netflix", "prime", "membership"],
    "Savings & Investments": ["sip", "mutual fund", "investment", "fd", "sip"],
    "Other": [],
}

# Example phrases to support TF-IDF similarity fallback
CATEGORY_EXAMPLES = {
    "Food & Dining": ["ordered pizza from zomato", "dinner at a restaurant", "coffee shop"],
    "Groceries": ["supermarket groceries", "bought vegetables and milk"],
    "Travel & Transport": ["uber to office", "flight booking to delhi", "bus ticket"],
    "Shopping": ["myntra purchase", "amazon order clothing"],
    "Bills & Utilities": ["paid electricity bill", "internet recharge", "gas bill payment"],
    "Entertainment": ["movie ticket booking", "netflix subscription renewal"],
    "Health & Medical": ["doctor consultation", "pharmacy medicines"],
    "Education": ["paid udemy course", "college tuition fees"],
    "Personal Care": ["salon haircut", "spa treatment"],
    "Subscriptions": ["spotify monthly fee", "prime membership"],
    "Savings & Investments": ["sip investment", "mutual fund purchase"],
    "Other": ["misc cash", "other expense"],
}

# Build TF-IDF vectorizer on examples
_all_examples = []
_example_labels = []
for cat, phrases in CATEGORY_EXAMPLES.items():
    for ph in phrases:
        _all_examples.append(ph)
        _example_labels.append(cat)
tfidf_vectorizer = TfidfVectorizer().fit(_all_examples or ["x"])
_example_vectors = tfidf_vectorizer.transform(_all_examples or ["x"])


def auto_detect_category(text, available=None):
    """
    Hybrid auto-detection:
      1) keyword rules (fast)
      2) TF-IDF similarity fallback
      3) last resort -> available[0] or 'Other'
    """
    s = (text or "").lower()
    if not s.strip():
        return "Other" if (not available or "Other" in available) else available[0]
    # rules
    for cat, keywords in AUTO_CATEGORIES.items():
        for kw in keywords:
            if kw in s:
                return cat if (not available or cat in available) else (available[0] if available else cat)
    # tfidf fallback
    try:
        v = tfidf_vectorizer.transform([s])
        sims = cosine_similarity(v, _example_vectors).flatten()
        idx = sims.argmax()
        cand = _example_labels[idx]
        if available and cand not in available:
            # choose next best that matches available
            for i in sims.argsort()[::-1]:
                if _example_labels[i] in available:
                    return _example_labels[i]
            return available[0]
        return cand
    except Exception:
        return available[0] if available else "Other"


# ========== Transaction DB helpers ==========
def add_transaction(user_id, date_str, amount, description="", category=None, t_type="Expense"):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO transactions (user_id, date, amount, description, category, type)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (user_id, date_str, float(amount), description, category, t_type),
    )
    conn.commit()
    conn.close()


def import_csv_for_user(user_id, file_obj):
    """
    CSV expected to have at least Date & Amount columns.
    Auto-categorizes each row.
    """
    df = pd.read_csv(file_obj)
    if not {"Date", "Amount"}.issubset(df.columns):
        raise ValueError("CSV must include 'Date' and 'Amount' columns")
    cats = load_categories()
    rows = 0
    for _, r in df.iterrows():
        d = pd.to_datetime(r["Date"], errors="coerce")
        if pd.isna(d):
            continue
        amt = float(r["Amount"])
        desc = ""
        if "Notes" in r and not pd.isna(r["Notes"]):
            desc = str(r["Notes"])
        elif "Description" in r and not pd.isna(r["Description"]):
            desc = str(r["Description"])
        cat = r.get("Category", None)
        if pd.isna(cat) or not cat:
            cat = auto_detect_category(desc, available=cats)
        ttype = r.get("Type", "Expense") if "Type" in r else "Expense"
        add_transaction(user_id, d.strftime("%Y-%m-%d"), amt, desc, cat, ttype)
        rows += 1
    return rows


def get_transactions(user_id):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM transactions WHERE user_id=? ORDER BY date DESC", conn, params=(user_id,)
    )
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["id", "user_id", "date", "amount", "description", "category", "type"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_categories():
    conn = get_conn()
    df = pd.read_sql_query("SELECT name FROM categories ORDER BY name", conn)
    conn.close()
    if df.empty:
        return ["Other"]
    return df["name"].tolist()


def get_category_totals(user_id):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT category, SUM(amount) as total FROM transactions WHERE user_id=? GROUP BY category",
        conn,
        params=(user_id,),
    )
    conn.close()
    if df.empty:
        return {}
    return {row["category"] if row["category"] else "Other": row["total"] for _, row in df.iterrows()}


def get_monthly_totals_df(user_id):
    conn = get_conn()
    df = pd.read_sql_query("SELECT date, amount, type FROM transactions WHERE user_id=?", conn, params=(user_id,))
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["month", "total"])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["amount"].sum().reset_index().sort_values("month")
    monthly.columns = ["month", "total"]
    return monthly


# ========== Goals helpers ==========
def add_goal(user_id, name, target_amount, target_date):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO goals (user_id, name, target_amount, current_amount, target_date) VALUES (?, ?, ?, ?, ?)",
        (user_id, name, float(target_amount), 0.0, target_date),
    )
    conn.commit()
    conn.close()


def get_goals(user_id):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM goals WHERE user_id=?", conn, params=(user_id,))
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["id", "user_id", "name", "target_amount", "current_amount", "target_date"])
    return df


# ========== Forecast helpers ==========
def linear_monthly_forecast(monthly_df, months_ahead=6):
    if monthly_df.empty or len(monthly_df) < 3:
        return pd.DataFrame(columns=["month", "forecast"])
    X = np.arange(len(monthly_df)).reshape(-1, 1)
    y = monthly_df["total"].values
    model = LinearRegression().fit(X, y)
    future_idx = np.arange(len(monthly_df), len(monthly_df) + months_ahead).reshape(-1, 1)
    preds = model.predict(future_idx)
    last_month = monthly_df["month"].max()
    months = [(last_month + pd.DateOffset(months=i + 1)).to_period("M").to_timestamp() for i in range(months_ahead)]
    return pd.DataFrame({"month": months, "forecast": preds})


def prophet_monthly_forecast(monthly_df, months_ahead=6):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not installed")
    dfp = monthly_df.rename(columns={"month": "ds", "total": "y"}).copy()
    dfp["ds"] = pd.to_datetime(dfp["ds"])
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(dfp)
    last = dfp["ds"].max()
    future = model.make_future_dataframe(periods=months_ahead, freq="M")
    pred = model.predict(future)
    pred = pred[["ds", "yhat"]].tail(months_ahead)
    pred["month"] = pred["ds"].dt.to_period("M").dt.to_timestamp()
    return pred[["month", "yhat"]].rename(columns={"yhat": "forecast"})


# ========== NLP quick parse for smart entry ==========
def parse_transaction_nlp(text):
    text = (text or "").strip().lower()
    today = datetime.today().date()
    # date parsing
    date_val = today
    if "today" in text:
        date_val = today
    elif "yesterday" in text:
        date_val = today - timedelta(days=1)
    else:
        m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})", text)
        if m:
            try:
                date_val = pd.to_datetime(m.group(), dayfirst=False).date()
            except Exception:
                date_val = today
    # amount parsing
    amt_m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    amount = float(amt_m.group(1)) if amt_m else 0.0
    # type
    if any(k in text for k in ["earned", "salary", "received", "credited"]):
        ttype = "Income"
    elif any(k in text for k in ["spent", "paid", "bought", "gave"]):
        ttype = "Expense"
    else:
        ttype = "Expense"
    return {"Date": date_val.strftime("%Y-%m-%d"), "Amount": amount, "Type": ttype, "Description": text}


# ========== Auth helpers ==========
def register_user(email, password):
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def verify_user(email, password):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email=? AND password=?", (email, password))
    r = c.fetchone()
    conn.close()
    return bool(r)


def get_user_id(email):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email=?", (email,))
    r = c.fetchone()
    conn.close()
    return r[0] if r else None


# ========== UI & Pages ==========
# Sidebar navigation + auth
with st.sidebar:
    st.markdown('<div class="header"><h3>💠 BudgetWise Pro</h3></div>', unsafe_allow_html=True)
    st.write("")
    menu = st.radio(
        "Navigate",
        ["Dashboard", "Add Transaction", "Smart Entry", "Upload CSV", "Forecast", "Goals", "Admin", "About"],
        index=0,
    )
    st.write("---")
    if "user_id" not in st.session_state:
        st.subheader("Login / Signup")
        email = st.text_input("Email", key="login_email")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Login"):
            if verify_user(email, pwd):
                st.session_state["user_id"] = get_user_id(email)
                st.success("Login successful")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")
        st.write("New? Create account below")
        new_email = st.text_input("New Email", key="reg_email")
        new_pwd = st.text_input("New Password", type="password", key="reg_pwd")
        if st.button("Sign Up"):
            ok = register_user(new_email, new_pwd)
            if ok:
                st.success("User created. Login now.")
            else:
                st.error("Could not register (email might exist).")
        st.write("---")
        st.info("Demo login: Afrin123@gmail.com / 123")
        st.stop()
    else:
        st.markdown(f"**Logged in:** `user_id={st.session_state['user_id']}`")
        if st.button("Logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.experimental_rerun()

user_id = st.session_state["user_id"]
available_categories = load_categories()

# ---------- Page implementations ----------


def page_add_transaction():
    st.header("Add Transaction")
    st.markdown("Quick add transaction. Category will be auto-suggested from description.")
    col1, col2 = st.columns([2, 1])
    with col1:
        date_in = st.date_input("Date", value=pd.to_datetime("today"))
        desc = st.text_input("Description", placeholder="e.g., Uber to office, Zomato order, Salary credited")
    with col2:
        amt = st.number_input("Amount (INR)", min_value=0.0, format="%.2f")
        # auto-detect
        suggested = auto_detect_category(desc, available=available_categories)
        sel_idx = available_categories.index(suggested) if suggested in available_categories else 0
        cat = st.selectbox("Category (auto-detected)", available_categories, index=sel_idx)
        tx_type = st.selectbox("Type", ["Expense", "Income"])
    if st.button("Add Transaction"):
        add_transaction(user_id, date_in.strftime("%Y-%m-%d"), float(amt), desc, cat, tx_type)
        st.success("Transaction added ✅")


def page_smart_entry():
    st.header("Smart Text Entry")
    st.markdown("Describe transactions in plain text. Example: 'Yesterday I spent 450 on groceries from bigbasket'")
    examples = [
        "Yesterday I spent 450 on groceries from bigbasket",
        "2023-06-05 earned 25000 salary from company",
        "Paid 320 for electricity bill on 12/07/2024",
        "Uber ride 120 to office",
        "Bought shirt from myntra 899",
    ]
    st.markdown("**Examples:** " + " | ".join(examples))
    txt = st.text_area("Transaction text", height=120)
    if st.button("Parse & Show"):
        parsed = parse_transaction_nlp(txt)
        parsed["Category"] = auto_detect_category(parsed.get("Description", ""), available_categories)
        st.json(parsed)
        if st.button("Add parsed transaction"):
            add_transaction(
                user_id, parsed["Date"], parsed["Amount"], parsed.get("Description", ""), parsed.get("Category"), parsed.get("Type", "Expense")
            )
            st.success("Parsed transaction added ✅")


def page_upload_csv():
    st.header("Upload CSV (auto-categorize)")
    st.markdown("CSV must include at least `Date` and `Amount`. Optional: Description/Notes, Category, Type.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            n = import_csv_for_user(user_id, uploaded)
            st.success(f"Imported {n} rows (auto-categorized)")
        except Exception as e:
            st.error("Import failed: " + str(e))
    st.markdown("---")
    st.info("Tip: place `Daily_Expenses_Extended.csv` in project folder and use Bulk-load from Admin.")


def page_dashboard():
    st.header("Dashboard")
    df = get_transactions(user_id)
    if df.empty:
        st.info("No transactions yet. Add or import to see insights.")
        return
    inc_total = df[df["type"].str.lower() == "income"]["amount"].sum()
    exp_total = df[df["type"].str.lower() == "expense"]["amount"].sum()
    balance = inc_total - exp_total
    savings_rate = (balance / inc_total * 100) if inc_total > 0 else (0 if exp_total > 0 else 100)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='card'><div class='kpi'>Total Income</div><div class='small-muted'>₹{inc_total:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='kpi'>Total Expense</div><div class='small-muted'>₹{exp_total:,.2f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='kpi'>Balance</div><div class='small-muted'>₹{balance:,.2f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'><div class='kpi'>Savings Rate</div><div class='small-muted'>{savings_rate:.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    # Category doughnut
    cat_totals = get_category_totals(user_id)
    if cat_totals:
        cat_df = pd.DataFrame(list(cat_totals.items()), columns=["Category", "Amount"]).sort_values("Amount", ascending=False)
        fig_pie = px.pie(cat_df, names="Category", values="Amount", hole=0.45, title="Spending by Category")
        st.plotly_chart(fig_pie, use_container_width=True)
    # Monthly income vs expense
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby(["month", "type"])["amount"].sum().reset_index()
    if not monthly.empty:
        fig_bar = px.bar(monthly, x="month", y="amount", color="type", barmode="group", title="Monthly Income vs Expense")
        fig_bar.update_xaxes(tickformat="%Y-%m")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Recent transactions")
    display_df = df[["date", "description", "amount", "category", "type"]].sort_values("date", ascending=False).head(20)
    display_df["amount"] = display_df["amount"].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(display_df, use_container_width=True)


def page_forecast():
    st.header("Forecast")
    monthly_df = get_monthly_totals_df(user_id)
    if monthly_df.empty or len(monthly_df) < 3:
        st.info("Not enough monthly data to forecast. Add transactions or bulk-load sample CSV.")
        return
    st.markdown("Historical monthly totals (last 24 months)")
    st.dataframe(monthly_df.tail(24).assign(total=lambda d: d["total"].apply(lambda v: f"₹{v:,.2f}")), use_container_width=True)
    months_ahead = st.selectbox("Forecast horizon (months)", [3, 6, 12], index=1)
    if PROPHET_AVAILABLE:
        method = st.radio("Method", ["Prophet (recommended)", "Linear (fallback)"], index=0)
    else:
        method = "Linear (prophet not available)"

    if st.button("Run Forecast"):
        if method.startswith("Prophet") and PROPHET_AVAILABLE:
            try:
                pred = prophet_monthly_forecast(monthly_df, months_ahead)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly_df["month"], y=monthly_df["total"], mode="lines+markers", name="Actual"))
                fig.add_trace(go.Scatter(x=pred["month"], y=pred["forecast"], mode="lines+markers", name="Forecast", line=dict(dash="dash")))
                fig.update_layout(title=f"Prophet Monthly Forecast ({months_ahead} months)", xaxis_title="Month", yaxis_title="Amount (INR)")
                st.plotly_chart(fig, use_container_width=True)
                st.success("Prophet forecast complete")
            except Exception as e:
                st.error("Prophet forecast failed: " + str(e))
        else:
            fut = linear_monthly_forecast(monthly_df, months_ahead)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_df["month"], y=monthly_df["total"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=fut["month"], y=fut["forecast"], mode="lines+markers", name="Linear Forecast", line=dict(dash="dash")))
            fig.update_layout(title=f"Linear Monthly Forecast ({months_ahead} months)", xaxis_title="Month", yaxis_title="Amount (INR)")
            st.plotly_chart(fig, use_container_width=True)
            st.success("Linear forecast displayed (fallback)")


def page_goals():
    st.header("Goals")
    st.markdown("Set financial goals and track manual progress.")
    with st.form("add_goal"):
        gname = st.text_input("Goal name")
        gtarget = st.number_input("Target amount (INR)", min_value=0.0, format="%.2f")
        gdate = st.date_input("Target date", value=date.today() + timedelta(days=90))
        submitted = st.form_submit_button("Add Goal")
        if submitted and gname and gtarget > 0:
            add_goal(user_id, gname, gtarget, gdate.strftime("%Y-%m-%d"))
            st.success("Goal created")
    st.markdown("---")
    df_goals = get_goals(user_id)
    if df_goals.empty:
        st.info("No goals yet. Create one above.")
        return
    for _, g in df_goals.iterrows():
        st.markdown(f"**{g['name']}** — Target: ₹{g['target_amount']:,.2f} — Due: {g['target_date']}")
        prog = (g["current_amount"] / g["target_amount"] * 100) if g["target_amount"] > 0 else 0
        st.progress(min(int(prog), 100))
        topup = st.number_input(f"Top-up amount for {g['name']}", min_value=0.0, format="%.2f", key=f"top_{g['id']}")
        if st.button(f"Top-up {g['name']}", key=f"btn_top_{g['id']}"):
            conn = get_conn()
            c = conn.cursor()
            c.execute("UPDATE goals SET current_amount = current_amount + ? WHERE id=? AND user_id=?", (float(topup), int(g["id"]), int(user_id)))
            conn.commit()
            conn.close()
            st.success("Goal updated")
            st.experimental_rerun()


def page_admin():
    st.header("Admin")
    st.markdown("Manage categories and view users. Use Bulk-load for sample CSV here.")
    conn = get_conn()
    c = conn.cursor()
    # Categories management
    st.subheader("Categories")
    cats = load_categories()
    st.write(", ".join(cats))
    new_cat = st.text_input("Add new category")
    if st.button("Add Category"):
        if new_cat.strip():
            try:
                c.execute("INSERT INTO categories (name) VALUES (?)", (new_cat.strip(),))
                conn.commit()
                st.success("Category added")
                st.experimental_rerun()
            except Exception as e:
                st.error("Add failed: " + str(e))
    # Bulk-load sample file (if present)
    st.markdown("---")
    st.subheader("Bulk-load sample CSV")
    if st.button("Bulk-load Daily_Expenses_Extended.csv (append)"):
        sample_path = "Daily_Expenses_Extended.csv"
        if not os.path.exists(sample_path):
            st.error("Sample CSV not found. Place Daily_Expenses_Extended.csv next to app.py")
        else:
            df_sample = pd.read_csv(sample_path)
            count = 0
            for _, r in df_sample.iterrows():
                try:
                    dstr = pd.to_datetime(r["Date"]).strftime("%Y-%m-%d")
                except Exception:
                    continue
                amt = float(r.get("Amount", 0.0))
                desc = r.get("Notes", r.get("Description", ""))
                cat = r.get("Category", None)
                if pd.isna(cat) or not cat:
                    cat = auto_detect_category(desc, available=available_categories)
                ttype = "Expense" if (r.get("Type") != "Income") else "Income"
                add_transaction(user_id, dstr, amt, desc, cat, ttype)
                count += 1
            st.success(f"Appended {count} rows")
    st.markdown("---")
    st.subheader("Users")
    users = pd.read_sql_query("SELECT id,email FROM users ORDER BY id DESC", get_conn())
    st.dataframe(users)


def page_about():
    st.header("About")
    st.markdown(
        """
    **BudgetWise Pro** — AI-enhanced budgeting & forecasting tool.

    - Auto-categorization (12 categories)
    - Prophet monthly forecasting (if installed)
    - Goals tracker, CSV import, Smart text entry
    - Built by Shaik Afrin Banu
    """
    )


# ---------- Router ----------
if menu == "Dashboard":
    page_dashboard()
elif menu == "Add Transaction":
    page_add_transaction()
elif menu == "Smart Entry":
    page_smart_entry()
elif menu == "Upload CSV":
    page_upload_csv()
elif menu == "Forecast":
    page_forecast()
elif menu == "Goals":
    page_goals()
elif menu == "Admin":
    page_admin()
else:
    page_about()

st.markdown("---")
st.write("Developed by Shaik Afrin Banu | Ravindra College of Engineering for Women | Infosys Springboard – Batch 3")
