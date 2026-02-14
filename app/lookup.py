"""
Structured Lookup Layer for Private AI.

Ingests CSV/Excel files into a per-client SQLite database,
enabling exact lookups for GL codes, rates, vendors, etc.
No LLM needed — pure table queries.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import re


# Base directory for client table databases (absolute, relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "client_tables"
TABLES_DIR.mkdir(exist_ok=True)


def get_db_path(client_id: str) -> Path:
    """Return the SQLite database path for a given client."""
    client_dir = TABLES_DIR / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    return client_dir / "tables.db"


def sanitize_table_name(filename: str) -> str:
    """Convert a filename into a safe SQLite table name."""
    name = Path(filename).stem
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_').lower()
    if not name:
        name = "table_data"
    return name


def ingest_table(client_id: str, filepath: Path, table_name: str = None) -> dict:
    """
    Ingest a CSV or Excel file into the client's SQLite database.

    Returns a dict with status info:
        {"success": bool, "table_name": str, "rows": int, "columns": list, "error": str|None}
    """
    db_path = get_db_path(client_id)

    if table_name is None:
        table_name = sanitize_table_name(filepath.name)

    try:
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath, dtype=str)
        elif filepath.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(filepath, dtype=str)
        else:
            return {
                "success": False,
                "table_name": table_name,
                "rows": 0,
                "columns": [],
                "error": f"Unsupported file type: {filepath.suffix}"
            }

        # Clean column names for SQLite compatibility
        df.columns = [
            re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).strip('_').lower()
            for col in df.columns
        ]

        # Strip whitespace from all string values
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        conn = sqlite3.connect(str(db_path))
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()

        return {
            "success": True,
            "table_name": table_name,
            "rows": len(df),
            "columns": list(df.columns),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "table_name": table_name,
            "rows": 0,
            "columns": [],
            "error": str(e)
        }


def list_tables(client_id: str) -> list:
    """List all tables available for a client."""
    db_path = get_db_path(client_id)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_table_info(client_id: str, table_name: str) -> dict:
    """Get column names and row count for a table."""
    db_path = get_db_path(client_id)
    if not db_path.exists():
        return {"columns": [], "rows": 0}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
    columns = [row[1] for row in cursor.fetchall()]

    cursor = conn.execute(f"SELECT COUNT(*) FROM '{table_name}'")
    rows = cursor.fetchone()[0]

    conn.close()
    return {"columns": columns, "rows": rows}


def search_table(client_id: str, table_name: str, search_term: str,
                 column: str = None) -> pd.DataFrame:
    """
    Search a table for rows matching the search term.

    If column is specified, searches only that column.
    Otherwise, searches ALL columns for a match.
    Returns a pandas DataFrame of matching rows.
    """
    db_path = get_db_path(client_id)
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))

    if column:
        query = f"SELECT * FROM '{table_name}' WHERE \"{column}\" LIKE ?"
        params = [f"%{search_term}%"]
    else:
        # Search across all columns
        info = get_table_info(client_id, table_name)
        if not info["columns"]:
            conn.close()
            return pd.DataFrame()

        conditions = " OR ".join(
            f"\"{col}\" LIKE ?" for col in info["columns"]
        )
        query = f"SELECT * FROM '{table_name}' WHERE {conditions}"
        params = [f"%{search_term}%"] * len(info["columns"])

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def query_exact(client_id: str, table_name: str, column: str,
                value: str) -> pd.DataFrame:
    """
    Exact match query on a specific column.
    Useful for GL code lookups, SCAC validation, etc.
    """
    db_path = get_db_path(client_id)
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    query = f"SELECT * FROM '{table_name}' WHERE \"{column}\" = ?"
    df = pd.read_sql_query(query, conn, params=[value])
    conn.close()
    return df


def delete_table(client_id: str, table_name: str) -> bool:
    """Delete a specific table from the client's database."""
    db_path = get_db_path(client_id)
    if not db_path.exists():
        return False

    conn = sqlite3.connect(str(db_path))
    conn.execute(f"DROP TABLE IF EXISTS '{table_name}'")
    conn.commit()
    conn.close()
    return True
