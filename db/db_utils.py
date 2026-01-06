# db/db_utils.py
# Fully corrected – includes import pandas as pd

import sqlite3
import os
import pandas as pd  # ← THIS WAS MISSING – NOW FIXED

DB_PATH = "data/sales.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Dedicated customers table
    c.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_name TEXT PRIMARY KEY
    )
    """)

    # Contracts table
    c.execute("""
    CREATE TABLE IF NOT EXISTS contracts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT,
        feature_id TEXT,
        feature_name TEXT,
        description TEXT,
        priority TEXT,
        FOREIGN KEY (customer_name) REFERENCES customers (customer_name)
    )
    """)

    # Releases table
    c.execute("""
    CREATE TABLE IF NOT EXISTS releases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT,
        feature_id TEXT,
        feature_name TEXT,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()

def store_contract_to_db(row: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Insert customer if not exists
    c.execute("INSERT OR IGNORE INTO customers (customer_name) VALUES (?)", (row["customer_name"],))
    
    # Insert contract features
    c.execute("""
    INSERT OR REPLACE INTO contracts 
    (customer_name, feature_id, feature_name, description, priority)
    VALUES (?, ?, ?, ?, ?)
    """, (
        row["customer_name"],
        row.get("feature_id"),
        row.get("feature_name"),
        row.get("description", ""),
        row.get("priority", "")
    ))
    
    conn.commit()
    conn.close()

def store_release_to_db(row: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
    INSERT OR REPLACE INTO releases 
    (customer_name, feature_id, feature_name, status)
    VALUES (?, ?, ?, ?)
    """, (
        row["customer_name"],
        row.get("feature_id"),
        row.get("feature_name"),
        row.get("status", "")
    ))
    
    conn.commit()
    conn.close()

def load_contracts_for_customer(customer_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT feature_id, feature_name, description, priority 
    FROM contracts 
    WHERE customer_name = ?
    """
    df = pd.read_sql_query(query, conn, params=(customer_name,))
    conn.close()
    return df if not df.empty else pd.DataFrame(columns=["feature_id", "feature_name", "description", "priority"])

def load_all_releases_for_customer(customer_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT feature_id, feature_name, status 
    FROM releases 
    WHERE customer_name = ?
    """
    df = pd.read_sql_query(query, conn, params=(customer_name,))
    conn.close()
    return df if not df.empty else pd.DataFrame(columns=["feature_id", "feature_name", "status"])