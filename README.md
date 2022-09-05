# Music Genre Classification: A Boilerplate ML Pipeline with MLflow and Weights & Biases

This example is a boilerplate for generating non-complex and reproducible ML pipelines with [MLflow](https://www.mlflow.org) and [Weights and Biases](https://wandb.ai/site). [Scikit-Learn](https://scikit-learn.org/stable/) is used as engine for the data preprocessing and modeling (concretely, a random forests model is trained). The pipeline is divided into the typical steps or components in a pipeline, carried out in order. 

The example comes originally from an exercise in [udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises), which I completed and extended with comments and some other minor features.

Table of contents:

- [Music Genre Classification: A Boilerplate ML Pipeline with MLflow and Weights & Biases](#music-genre-classification-a-boilerplate-ml-pipeline-with-mlflow-and-weights--biases)
    - [Overview of Boilerplate Project Structure](#overview-of-boilerplate-project-structure)
    - [Dependencies](#dependencies)
    - [How to Run: Pipeline Creation and Deployment](#how-to-run-pipeline-creation-and-deployment)
      - [Run the Pipeline to Generate the Inference Artifacts](#run-the-pipeline-to-generate-the-inference-artifacts)
      - [Deployment: Use the Inference Artifacts for Performing Predictions](#deployment-use-the-inference-artifacts-for-performing-predictions)
    - [Interesting Links](#interesting-links)
    - [Authorship](#authorship)

### Overview of Boilerplate Project Structure

The file structure of the folder is the following:

```
.
├── MLproject
├── README.md
├── check_data
│   ├── MLproject
│   ├── conda.yml
│   ├── conftest.py
│   └── test_data.py
├── conda.yml
├── config.yaml
├── dataset
│   └── genres_mod.parquet
├── download
│   ├── MLproject
│   ├── conda.yml
│   └── download_data.py
├── evaluate
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── main.py
├── preprocess
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── random_forest
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── segregate
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
└── test_inference
    ├── README.md
    └── test_inference.py
```

The ML problem consists in classifying music song genres depending on 

The most important high-level files are `config.yaml` and `main.py`; they contain the parameters and the main pipeline execution order, respectively. Each component or pipeline step has its own project sub-folder, with their `MLproject` and `conda.yaml` files, for `mlflow` and conda environment configuration, respectively.

Pipeline steps or components:

1. `download/`
    - A parquet file of songs and their attributes is downloaded from a URL; the songs need to be classified according to their genre.
    - The dataset it uploaded to Weights and Biases as an artifact.
2. `preprocess/`
    - Raw dataset artifact is downloaded and preprocessed: missing values imputed and duplicates removed; additionally, a new feature `text_feature` is created.
3. `check_data/`
    - Data validation: pre-processed dataset is checked using `pytest`.
    - In the dummy example, the reference and sample datasets are the same, and only deterministic tests are carried out, but we could have used a reference dataset for non-deterministic tests.
4. `segregate/`
    - Train/test split is done and the two splits are uploaded as artifacts.
5. `random_forest/`
    - Component/step with which a random forest model is defined and trained.
    - The training split is subdivided to train/validation.
    - The model is packed in a pipeline which contains data preprocessing and the model itself
        - The data preprocessing differentiates between numerical/categorical/NLP columns with `ColumnTransformer` and performs imputations, reshapings, feature extraction (TDIDF) and scaling.
        - The model configuration is achieved via a temporary yaml file created in the `main.py` using the parameters from `config.yaml`.
    - That inference pipeline is exported and uploaded as an artifact.
    - Performance metrics (AUC) and images (confusion matrix, feature importances) are generated and uploaded as artifacts.
6. `evaluate/`
    - The artifacts related to the test split and the inference pipeline are downloaded and used to compute the metrics with the test dataset.

Obviously, not all steps need to be carried out every time; to that end, with have the parameter `main.execute_steps` in the `config.yaml`. We can override it when calling `mlflow run`.

Note that there are some other folder/steps that come before or after the inference pipeline; each of them has an explanatory file that extends the current `README.md`:

- `data_analysis/`: simple Exploratory Data Analysis (EDA) and Feature Engineering (FE) are performed, as well as data modeling with cross validation to find the optimum hyperparameters. In this folder, the step from a research environment (Jupyter Notebook) to a development environment is shown. The focus doesn't lie on the EDA / FE / Modeling parts, but rather on the transformation of the code for production.
- `test_inference/`: the exported inference pipeline artifact is deployed to production using different approaches.

### Dependencies

All project and component dependencies are specified in the `conda.yaml` files.

For the ML operations, you require the following tools:

- [MLflow](https://www.mlflow.org): management of step parametrized step executions.
- [Weights and Biases](https://wandb.ai/site): tracking of experiments, artifacts, metrics, etc.

In order to install those tools:

```bash
# Create an environment
# conda create --name ds-env python=3.8 mlflow jupyter pandas matplotlib requests -c conda-forge
# ... or activate an existing one:
conda activate ds-env

# Install MLflow packages
conda install mlflow requests -c conda-forge

# Make sure pip is pointing to the pip in the conda environment
conda install pip
which pip

# Install Weights and Biases through pip
pip install wandb

# Log in to wandb
wandb login
# Log in on browser if not done
# Go to provided URL: https://wandb.ai/authorize
# Copy and paste API key on Terminal, as requested
# Done!
```

Note that [hydra](https://hydra.cc/docs/intro/) is also employed in the project; the dependency is resolved with the `conda.yaml` environment configuration files.

### How to Run: Pipeline Creation and Deployment

This section deals with the creation and deployment of the inference pipeline; if you are interested in the data analysis that precedes it, please check the dedicated folder [data_analisis](data_analisis/README.md).

First, we need to run the entire pipeline (all steps) at least once (locally or remotely) to generate all the artifacts. For that, we need to be logged with our WandB account. After that, we can perform online/offline predictions with MLflow. The correct model version & co. needs to be checked on the WandB web interface.

#### Run the Pipeline to Generate the Inference Artifacts

**To run the code locally**:

```bash
cd path-to-main-mlflow-file
# All steps are executed, in the order defined in main.py
mlflow run .

# Run selected steps: Download dataset and preprocess it
# The order is given by main.py
mlflow run . -P hydra_options="main.execute_steps='download,preprocess,check_data,segregate'"

# Run selected steps: Just re-generate the inference pipeline and evaluate it
# The order is given by main.py
mlflow run . -P hydra_options="main.execute_steps='random_forest,evaluate'"

# For production: 
# Change the name of the project for production
mlflow run . -P hydra_options="main.project_name='music_genre_classification_prod'"
```

Since the repository is publicly released, anyone can **run the code remotely** as follows:

```bash
# Go to a new empty folder
cd new-empty-folder

# General command
mlflow run -v [commit hash or branch name] [URL of your Github repo]

# Concrete command for the exercise in section 6.1
# We point to the commit hash of tag 0.0.1
# Note: currently, we cannot put tag names in -v
mlflow run -v 82f17d94e0800811e81f4d55c0442d3189ed0a63 git@github.com:mxagar/music_genre_classification.git

# Project name changed
# We point to the branch main; usually, we should point to a branch like master / stable, etc.
# Note: currently, we cannot put tag names in -v
mlflow run git@github.com:mxagar/music_genre_classification.git -v main -P hydra_options="main.project_name=remote_execution"
```

#### Deployment: Use the Inference Artifacts for Performing Predictions

See [test_inference/README.md](test_inference/README.md) for more information on:

- How to download and use the inference pipeline to perform batch predictions via CLI.
- How to serve the inference pipeline as a REST API.
- How to create a docker image that serves the REST API.

Before serving or deploying anything, we need to have run the entire pipeline at least once, as explained above, e.g., locally:

```bash
mlflow run .
```

### Interesting Links

- This repository doesn't focus on the techniques for data processing and modeling; if you are interested in those topics, you can visit my  [Guide on EDA, Data Cleaning and Feature Engineering](https://github.com/mxagar/eda_fe_summary).
- If you are interested in more MLOps-related content, you can visit my notes on the [Udacity Machine Learning DevOps Engineering Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821): [mlops_udacity](https://github.com/mxagar/mlops_udacity).


### Authorship

Mikel Sagardia, 2022.  
No guarantees.

I you find this repository useful and use it, please link to the original source.