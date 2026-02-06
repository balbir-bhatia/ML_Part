import pandas as pd
from config.db_config import get_pg_engine

def load_postgres_tables(
    customer_table: str = "customers",
    merchant_table: str = "merchants",
    transaction_table: str = "transactions",
    limit: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = get_pg_engine()
    lim = f" LIMIT {limit}" if limit else ""
    customers = pd.read_sql(f"SELECT * FROM {customer_table}{lim};", engine)
    merchants = pd.read_sql(f"SELECT * FROM {merchant_table}{lim};", engine)
    transactions = pd.read_sql(f"SELECT * FROM {transaction_table}{lim};", engine)
    return customers, merchants, transactions