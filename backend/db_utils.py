import sqlite3, os, pandas as pd
from .categorize_utils import guess_category

DB_PATH = os.path.join(os.path.dirname(__file__), "budget_tool.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash BLOB NOT NULL
    );""")
    c.execute("""CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        amount REAL NOT NULL,
        category TEXT,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );""")
    conn.commit(); conn.close()

def add_transaction(user_id:int, date:str, amount:float, description:str=''):
    category = guess_category(description)
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('INSERT INTO transactions (user_id,date,amount,category,description) VALUES (?,?,?,?,?)', (user_id,date,amount,category,description))
    conn.commit(); conn.close()
    return True

def get_transactions(user_id:int):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT date, description, amount, category FROM transactions WHERE user_id=? ORDER BY date DESC', conn, params=(user_id,))
    conn.close()
    return df

def import_csv_for_user(user_id:int, file_like):
    df = pd.read_csv(file_like)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get('date') or cols.get('transaction_date') or cols.get('timestamp') or None
    amount_col = cols.get('amount') or cols.get('amount_in_inr') or cols.get('amt') or cols.get('total') or None
    desc_col = cols.get('description') or cols.get('desc') or None
    if date_col is None or amount_col is None:
        raise ValueError('CSV must contain date and amount columns (case-insensitive)')
    rows = []
    for _, r in df.iterrows():
        try:
            date_val = pd.to_datetime(r[date_col], dayfirst=False, errors='coerce')
            date_str = date_val.strftime('%Y-%m-%d') if not pd.isna(date_val) else str(r[date_col])
        except Exception:
            date_str = str(r[date_col])
        try:
            amount = float(r[amount_col])
        except Exception:
            amount = 0.0
        description = str(r[desc_col]) if desc_col and not pd.isna(r.get(desc_col)) else ''
        category = guess_category(description)
        rows.append((user_id, date_str, amount, category, description))
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.executemany('INSERT INTO transactions (user_id,date,amount,category,description) VALUES (?,?,?,?,?)', rows)
    conn.commit(); conn.close()
    return len(rows)

def get_category_totals(user_id:int):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT category, SUM(amount) FROM transactions WHERE user_id=? GROUP BY category', (user_id,))
    rows = c.fetchall(); conn.close()
    return {r[0] if r[0] is not None else 'Others': float(r[1]) for r in rows}

def get_monthly_totals(user_id:int):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT substr(date,1,7) as ym, SUM(amount) FROM transactions WHERE user_id=? GROUP BY ym ORDER BY ym", (user_id,))
    rows = c.fetchall(); conn.close()
    labels = [r[0] for r in rows]; totals=[float(r[1]) for r in rows]
    return labels, totals
