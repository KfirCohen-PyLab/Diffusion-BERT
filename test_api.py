import requests

url = 'http://localhost:5000/predict'
data = {
    'text': 'This movie is great! I really enjoyed watching it.'
}

response = requests.post(url, json=data)
print(response.json())