import pandas as pd

def merge_core(customers: pd.DataFrame, merchants: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    # Normalize keys to string
    for df, key in [(customers, "customer_id"), (merchants, "merchant_id"), (transactions, "customer_id"), (transactions, "merchant_id")]:
        if key in df.columns:
            df[key] = df[key].astype(str)

    tx = transactions.copy()
    # Join user attributes
    tx = tx.merge(
        customers.add_prefix("user_").rename(columns={"user_customer_id": "customer_id"}),
        on="customer_id", how="left"
    )
    # Join merchant attributes (context only)
    tx = tx.merge(
        merchants.add_prefix("merchant_").rename(columns={"merchant_merchant_id": "merchant_id"}),
        on="merchant_id", how="left"
    )
    return tx