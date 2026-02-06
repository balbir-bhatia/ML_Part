import os
import pandas as pd
from config.settings import settings

def load_csvs(
    customers_path: str | None = None,
    merchants_path: str | None = None,
    transactions_path: str | None = None
):
    customers_path = customers_path or os.path.join("data","raw","customers.csv")
    merchants_path = merchants_path or os.path.join("data","raw","merchants.csv")
    transactions_path = transactions_path or os.path.join("data","raw","transactions.csv")

    customers = pd.read_csv(customers_path)
    merchants = pd.read_csv(merchants_path)
    transactions = pd.read_csv(transactions_path)
    return customers, merchants, transactions