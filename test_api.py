import requests

# Test with provided data
response = requests.post(
    'http://localhost:8000/recommend',
    json={
        'customer_id': 'CUST1001',
        'recent_service_ids': ['MER1002', 'MER1004'],
        'top_k': 5
    }
)

print(f'Status Code: {response.status_code}')
print('Response Body:')
print(response.json())