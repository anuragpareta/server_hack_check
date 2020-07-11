import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'X_1':0,'X_2':36,'X_3':37,'X_4':6,'X_5':3,'X_6':9,'X_7':18,'X_8':1,'X_9':6,'X_10':1,'X_11':303,'X_12':0,'X_13':112,'X_14':62,'X_15':34,})

print(r.json())