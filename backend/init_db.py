import sqlite3, os, bcrypt

DB_PATH = os.path.join(os.path.dirname(__file__), "budget_tool.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    demo_email = "Afrin123@gmail.com"
    demo_password = "123"
    c.execute("SELECT id FROM users WHERE email=?", (demo_email,))
    if not c.fetchone():
        pw_hash = bcrypt.hashpw(demo_password.encode('utf-8'), bcrypt.gensalt())
        c.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (demo_email, pw_hash))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print('DB initialized at', DB_PATH)
