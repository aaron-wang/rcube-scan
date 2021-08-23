import json

with open("config.json") as f:
    data = json.load(f)

for i in data['colors']:
    print(i['name'])
    print(i['lower_bound'])

for key in data['bob']:
    print(key)