import pandas as pd
import json

def parse_items(items):
    if items is None or (isinstance(items, float) and pd.isna(items)):
        return []
    if isinstance(items, list):
        return items
    if isinstance(items, dict):
        return [items]
    if isinstance(items, str):
        try:
            val = json.loads(items)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return [val]
        except Exception:
            return []
    return []