import sqlite3
import os
import re
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join("data", "users.db")

def validate_account(account):
    """Validate if the account is a valid email or phone number."""
    # Email regex
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    # Phone regex (Chinese mobile numbers: 11 digits starting with 1)
    phone_regex = r'^1[3-9]\d{9}$'
    
    if re.match(email_regex, account):
        return True, "email"
    elif re.match(phone_regex, account):
        return True, "phone"
    return False, None

def init_db():
    """Initialize the users database and migrate if necessary."""
    if not os.path.exists("data"):
        os.makedirs("data")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    else:
        # Check if 'username' column exists, if so rename to 'account'
        cursor.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'username' in columns and 'account' not in columns:
            try:
                cursor.execute("ALTER TABLE users RENAME COLUMN username TO account")
            except Exception as e:
                print(f"Migration warning: {e}")
        
        # Check if 'role' column exists
        if 'role' not in columns:
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
            except Exception as e:
                print(f"Migration warning (role): {e}")

    conn.commit()
    conn.close()

def register_user(account, password):
    """Register a new user with account (email/phone)."""
    # Validate account format
    is_valid, _ = validate_account(account)
    if not is_valid:
        return False, "账号格式不正确，请输入有效的邮箱或手机号"

    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if this is the first user
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        role = 'admin' if count == 0 else 'user'
        
        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (account, password, role) VALUES (?, ?, ?)", (account, hashed_password, role))
        conn.commit()
        return True, "注册成功"
    except sqlite3.IntegrityError:
        return False, "该账号已被注册"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def login_user(account, password):
    """Authenticate a user."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, password, role FROM users WHERE account = ?", (account,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        return True, {"user_id": str(user[0]), "role": user[2]}  # Return dict with user_id and role
    return False, "账号或密码错误"

def delete_user(user_id):
    """Delete a user by ID."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            return False, "用户不存在"
            
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return True, "删除成功"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def reset_password(user_id, new_password):
    """Reset user password."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            return False, "用户不存在"
            
        hashed_password = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
        conn.commit()
        return True, "密码重置成功"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

