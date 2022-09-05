# Using the Inference Artifact for Performing Predictions

This file and folder contain information on:

- How to download and use the inference pipeline to perform batch predictions via CLI.
- How to serve the inference pipeline as a REST API.
- How to create a docker image that serves the REST API.

Before serving or deploying anything, we need to have run the entire pipeline at least once, as explained above, e.g., locally:

```bash
mlflow run .
```

### Offline or Batch Inference

An offline or batch inference is characterized by many samples inferred, optimizing for throughput.

Example workflow of how to carry it out:

```bash
cd .../new_empty_folder
# In our case: cd test_inference

# Get the artifact
# We can check in the web interface the address of the desired artifact:
# Go to project, artifacts, model; make sure to mark it with prod tag for aligning with conventions
# --root: directory to which the artifact is downloaded, model/
# General call (user is optinal if logged in):
# wandb artifact get [<user>/]<project_name>/<inference_artifact>:<tag> --root model
wandb artifact get datamix-ai/music_genre_classification_prod/model_export:prod --root model

# The artifact should be in the specified folder: model/
# The artifact consists of
# - the MLflow file
# - the automatically generated environment file
# - the example JSON with the example samples we used during creation
# - a pickle of the model or the equivalent serialized object
cd model
tree
# .
# ├── MLmodel
# ├── conda.yaml
# ├── input_example.json
# └── model.pkl

# Predict offline/batch
# mlflow models: mlflow deployment API; predict = offline serving
# -t: format, json or csv
# -i: input file in specified format
# -m model: folder where the artifact should be
mlflow models predict -t json -i model/input_example.json -m model
# The conda env is created in the first call
# We get
# ["Rap", "RnB"]

# Another example
# With this, we get a prediction for each row in data_test.csv
wandb artifact get music_genre_classification_prod/model_export:prod --root model
wandb artifact get music_genre_classification_prod/data_test.csv:latest
mlflow models predict -t csv -i artifacts/data_test.csv/v0/data_test.csv -m model

```

### Online or Realtime Inference 

An online or realtime inference is characterized by one/few samples at a time, optimizing for speed/latency. To apply it, first, we need to get the model artifact as we did before; then, we call `mlflow models serve` instead of `mlflow models predict`. The command `mlflow models serve` creates a microservice with a REST API which we can access in multiple ways, e.g., with Python using `resquests`.

The service creation with the REST API:

```bash
# Get the inference/model artifact
cd .../new_empty_folder # test_inference
# wandb artifact get [<user>/]<project_name>/<inference_artifact>:<tag> --root model
wandb artifact get music_genre_classification_prod/model_export:prod --root model

# Predict online/realtime
# mlflow models: MLflow deployment API; serve = online
# -m model: folder where the artifact should be
mlflow models serve -m model &
# We get the message:
# Listening at: http://127.0.0.1:5000
# That is, we can post JSONs as before to the following interface and get predictions:
# http://127.0.0.1:5000/invocations
# IMPORTANT NOTE: use 127.0.0.1 and not localhost,
# because sometimes it is not converted correctly by the requests package
```

Accessing the REST API with python: We create a script `test_inference.py` and execute it as `python test_inference.py`; the content of `test_inference.py`:

```python
import requests
import json

with open("model/input_example.json") as fp:
    data = json.load(fp)

print(data)

# JSON has an issue with fields that contain NaN/nan
# They need to be converted to a null string in the JSON file itself (nan -> null)
# or to None in Python.
# One possible solution in Python is the following:
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
```

### Docker Image

We can create a Docker image which can be instantiated on a Cloud platform (e.g., AWS) as a container. If we expose its port 5000, then the model is listening to the world!

More information: [mlflow-models-build-docker](https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker).

First, make sure you have Docker installed and running, and that you downloaded the inference artifact with `wandb artifact get`. Then, create the image and test it:

```bash
# Build a generic Docker image:
# Specify the image --name and the folder of the model as downloaded (-m)
mlflow models build-docker -m model --name "music_genre_classification"

# Mount the model stored in './model' and serve it
docker run --rm -p 5000:8080 -v ./model:/opt/ml/model "music_genre_classification"

# Now, the server is listening under
# [url to the deployed machine]:5000/invocations
```

Unfortunately, I had issues with the Java certifications suing my Apple M1.

One helpful link to write the Dockerfile manually could be this one (not tried): [Using MLFlow and Docker to Deploy Machine Learning Models](https://medium.com/@paul.bendevis/using-mlflow-and-docker-to-deploy-machine-learning-models-4f7888005e24).
