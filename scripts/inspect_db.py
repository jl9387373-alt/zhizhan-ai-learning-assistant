import sqlite3
import os
import sys

def inspect_db(db_path):
    print(f"\n=== Inspecting Database: {db_path} ===")
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found.")
        
        for table_name in tables:
            table = table_name[0]
            print(f"\n--- Table: {table} ---")
            
            # Get columns
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [info[1] for info in cursor.fetchall()]
            print(f"Columns: {', '.join(columns)}")
            
            # Get rows
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            if not rows:
                print("No data.")
            else:
                for row in rows:
                    print(row)
                    
        conn.close()
    except Exception as e:
        print(f"Error inspecting {db_path}: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    # Inspect users.db
    inspect_db(os.path.join(data_dir, "users.db"))
    
    # Inspect chat_memory.db if exists
    chat_db = os.path.join(data_dir, "chat_memory.db")
    if os.path.exists(chat_db):
        inspect_db(chat_db)
