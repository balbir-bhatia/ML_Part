import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MODELS = os.path.join(BASE_DIR, 'data', 'models')
DATA_FEATURES = os.path.join(BASE_DIR, 'data', 'features')
API_HOST = '0.0.0.0'
API_PORT = 8000