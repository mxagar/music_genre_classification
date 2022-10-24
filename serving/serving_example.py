import requests
import json

with open("model/input_example.json") as fp:
    data = json.load(fp)

print(data)

# JSON has an issue with fields that contain NaN/nan
# They need to be converted to a null string in the JSON file itself (nan -> null)
# or to None in Python.
# One option should be to use YAML and yaml.safe_load()?
# Another possible solution in Python is the following:
import math
import numpy as np

def to_none(val):
    if not isinstance(val,str):
        if math.isnan(val):
            return None
    return val

def fix_nan(arr):
    arr_new = arr
    # Assuming 2D tables with 1D objects in cells, i.e., no lists in cells
    for row in range(len(arr)):
        for col in range(len(arr[0])):
            arr_new[row][col] = to_none(arr_new[row][col])
    return arr_new

data['data'] = fix_nan(data['data'])

# NaN has been converted to None
print(data)

# Get the prediction/inference through the REST API
# Do not use localhost, use 127.0.0.1, otherwise we could have issues (403 response)
results = requests.post("http://127.0.0.1:5000/invocations", json=data)

# We should get a response 200: correct
print(results)

# Result
print(results.json()) # ['Rap', 'RnB']