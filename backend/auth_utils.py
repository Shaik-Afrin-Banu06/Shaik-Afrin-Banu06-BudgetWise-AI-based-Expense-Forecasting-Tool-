import sqlite3, os, bcrypt
DB_PATH = os.path.join(os.path.dirname(__file__), "budget_tool.db")

def ensure_users_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash BLOB NOT NULL
    );""")
    conn.commit(); conn.close()

ensure_users_table()

def register_user(email, password):
    try:
        pw_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute('INSERT INTO users (email, password_hash) VALUES (?,?)', (email, pw_hash))
        conn.commit(); conn.close()
        return True
    except Exception:
        return False

def verify_user(email, password):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE email=?', (email,))
    r = c.fetchone(); conn.close()
    if not r:
        return False
    try:
        return bcrypt.checkpw(password.encode('utf-8'), r[0])
    except Exception:
        return False

def get_user_id(email):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT id FROM users WHERE email=?', (email,))
    r = c.fetchone(); conn.close()
    return r[0] if r else None
