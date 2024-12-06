import requests

url = 'http://127.0.0.1:5000/predict'


o3_values = [0.013, 0.009, 0.011]

data = {'o3_values': o3_values}

response = requests.post(url, json=data)

if response.status_code == 200:
    predictions = response.json()['predictions']
    print(f'Next day: {predictions[0]:.4f}')
else:
    print(f'Error: {response.status_code}')
    print(response.json())
