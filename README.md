# Music Genre Classification: A Boilerplate Reproducible ML Pipeline with MLflow and Weights & Biases

This project is a boilerplate for generating non-complex and reproducible ML pipelines with [MLflow](https://www.mlflow.org) and [Weights and Biases](https://wandb.ai/site). [Scikit-Learn](https://scikit-learn.org/stable/) is used as engine for the data preprocessing and modeling (concretely, a random forests model is trained), and [conda](https://docs.conda.io/en/latest/) as environment management system. The pipeline is divided into the typical steps or components in a pipeline, carried out in order. 

The example comes originally from an exercise in the repository [udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises), which I completed and extended with comments and some other minor features.

The used dataset is a modified version of the [songs in Spotify @ Kaggle](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify): in contains 40k+ song entries, each with 12 features, and the target variable is the genre they belong to. More information on the dataset can be found in the folder `data_analysis`, which is not part of the inference pipeline.

Table of contents:

- [Music Genre Classification: A Boilerplate Reproducible ML Pipeline with MLflow and Weights & Biases](#music-genre-classification-a-boilerplate-reproducible-ml-pipeline-with-mlflow-and-weights--biases)
  - [Overview of Boilerplate Project Structure](#overview-of-boilerplate-project-structure)
    - [Data Analysis and Serving](#data-analysis-and-serving)
  - [How to Use this Guide](#how-to-use-this-guide)
  - [Dependencies](#dependencies)
  - [How to Run This: Pipeline Creation and Deployment](#how-to-run-this-pipeline-creation-and-deployment)
    - [Run the Pipeline to Generate the Inference Artifacts](#run-the-pipeline-to-generate-the-inference-artifacts)
    - [Deployment: Use the Inference Artifacts for Performing Predictions](#deployment-use-the-inference-artifacts-for-performing-predictions)
  - [Notes on How MLflow and Weights & Biases Work](#notes-on-how-mlflow-and-weights--biases-work)
    - [Component Script Structure](#component-script-structure)
    - [Tips and Tricks](#tips-and-tricks)
  - [Improvements, Next Steps](#improvements-next-steps)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## Overview of Boilerplate Project Structure

The file structure of the root folder is the following:

```
.
├── README.md
├── MLproject
├── config.yaml
├── conda.yml
├── main.py
├── dataset/
│   └── genres_mod.parquet
├── data_analysis/
│   ├── MLproject
│   ├── conda.yml
│   ├── EDA_Tracking.ipynb
│   ├── Modeling.ipynb
│   └── README.md
├── get_data/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── preprocess/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── check_data/
│   ├── MLproject
│   ├── conda.yml
│   ├── conftest.py
│   └── test_data.py
├── segregate/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── train_random_forest/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── evaluate/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
└── serving/
    ├── README.md
    ├── serving_example.py
    └── ...
```

![Reproducible ML Pipeline: Generic Workflow](assets/Reproducible_Pipeline.png)

The most important high-level files are `config.yaml` and `main.py`; they contain the parameters and the main pipeline execution order, respectively. Each component or pipeline step has its own project sub-folder, with their `MLproject` and `conda.yaml` files, for `mlflow` and conda environment configuration, respectively.

Pipeline steps or components:

1. [`get_data/`](get_data)
    - A parquet file of songs and their attributes is downloaded from a URL; the songs need to be classified according to their genre.
    - The dataset it uploaded to Weights and Biases as an artifact.
2. [`preprocess/`](preprocess)
    - Raw dataset artifact is downloaded and preprocessed: missing values imputed and duplicates removed; additionally, a new feature `text_feature` is created.
3. [`check_data/`](check_data)
    - Data validation: pre-processed dataset is checked using `pytest`.
    - In the dummy example, the reference and sample datasets are the same, and only deterministic tests are carried out, but we could have used a reference dataset for non-deterministic tests.
4. [`segregate/`](segregate)
    - Train/test split is done and the two splits are uploaded as artifacts.
5. [`train_random_forest/`](random_forest)
    - Component/step with which a random forest model is defined and trained.
    - The training split is subdivided to train/validation.
    - The model is packed in a pipeline which contains data preprocessing and the model itself
        - The data preprocessing differentiates between numerical/categorical/NLP columns with `ColumnTransformer` and performs imputations, re-shapings, text feature extraction (TDIDF) and scaling.
        - The model configuration is achieved via a temporary YAML file created in the `main.py` using the parameters from `config.yaml`.
    - That inference pipeline is exported and uploaded as an artifact.
    - Performance metrics (AUC) and images (confusion matrix, feature importances) are generated and uploaded as artifacts.
6. [`evaluate/`](evaluate)
    - The artifacts related to the test split and the inference pipeline are downloaded and used to compute the metrics with the test dataset.

Obviously, not all steps need to be carried out every time; to that end, with have the parameter `main.execute_steps` in the `config.yaml`. We can override it when calling `mlflow run`, as shown in the section [How to Run This](#how-to-run-pipeline-creation-and-deployment).

### Data Analysis and Serving

There are two stand-alone folders/steps that are not part of the inference pipeline; each of them has an explanatory file that extends the current `README.md`:

- [`data_analysis`](data_analysis/README.md): simple Exploratory Data Analysis (EDA), Data Cleaning and Feature Engineering (FE) are performed, as well as data modeling with cross validation to find the optimum hyperparameters. In this folder, the step from a research environment (Jupyter Notebook) to a development environment is shown. The focus doesn't lie on the EDA / FE / Modeling parts, but rather on the transformation of the code for production; if you are interested in the former, you can visit my [Guide on EDA, Data Cleaning and Feature Engineering](https://github.com/mxagar/eda_fe_summary).
- [`serving`](serving/README.md): the exported inference pipeline artifact is deployed to production using different approaches.

## How to Use this Guide

1. Have a look at [`data_analysis`](data_analysis/README.md): a simple data preprocessing (+ EDA) is performed to understand the dataset.
2. Then, you can check the pipeline managed with [MLflow](https://www.mlflow.org) and [Weights and Biases](https://wandb.ai/site): Follow the notes on the [Overview](#overview-of-boilerplate-project-structure) and run the tracked pipeline as explained in [How to Run: Pipeline Creation and Deployment](#how-to-run-pipeline-creation-and-deployment); make sure you have installed the [dependencies](#dependencies).

## Dependencies

You need to have:

- [Conda](https://docs.conda.io/en/latest/): environment management.
- [MLflow](https://www.mlflow.org): management of step parametrized step executions.
- [Weights and Biases](https://wandb.ai/site): tracking of experiments, artifacts, metrics, etc.

Note that each ML component has its specific `conda` environment, each one specified in their respective `conda.yaml` files. On th eother hand, you can follow this shirt recipe to set up the main environment from which everything is launched:

```bash
# Create a basic environment: we use this environment to launch everything
conda create --name ds-env python=3.8 numpy pandas matplotlib scikit-learn requests jupyter -c conda-forge

# Activate environment
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

It is possible to automate all that with a `conda.yaml`, but I wanted to leave all steps to make clear what's going on, since this is a boilerplate :wink:

Note that [hydra](https://hydra.cc/docs/intro/) is also employed in the project; the dependency is resolved with the `conda.yaml` environment configuration files.

## How to Run This: Pipeline Creation and Deployment

This section deals with the creation and deployment of the inference pipeline; if you are interested in the data analysis that precedes it, please check the dedicated folder [`data_analisis`](data_analisis/README.md).

First, we need to run the entire pipeline (all steps) at least once (locally or remotely) to generate all the artifacts. For that, we need to be logged with our WandB account. After that, we can perform online/offline predictions with MLflow. The correct model version & co. needs to be checked on the WandB web interface.

### Run the Pipeline to Generate the Inference Artifacts

In order to **run the code locally**:

```bash
cd path-to-main-mlflow-file
# All steps are executed, in the order defined in main.py
mlflow run .

# Run selected steps: Download dataset and preprocess it
# The order is given by main.py
mlflow run . -P hydra_options="main.execute_steps='get_data,preprocess,check_data,segregate'"

# Run selected steps: Just re-generate the inference pipeline and evaluate it
# The order is given by main.py
mlflow run . -P hydra_options="main.execute_steps='train_random_forest,evaluate'"

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

### Deployment: Use the Inference Artifacts for Performing Predictions

See [`serving/README.md`](serving/README.md) for more information on:

- How to download and use the inference pipeline to perform batch predictions via CLI.
- How to serve the inference pipeline as a REST API.
- How to create a docker image that serves the REST API.

Before serving or deploying anything, we need to have run the entire pipeline at least once, as explained above, e.g., locally:

```bash
mlflow run .
```

## Notes on How MLflow and Weights & Biases Work

### Component Script Structure

```python
# Imports
# ...

# Logger
logging.basicConfig(...)
logger = logging.getLogger()

# Component function
# args contains all necessary variables
def go(args):
    # Instantiate W&B run in a context
    # or, if preferred, without a context
    # run = wandb.init(...)
    # IMPORTANT: set project="my_project"
    # to share artifacts between different components
    with wandb.init(project="my_project", ...) as run:

        # Upload configuration params in a dictionary,
        # e.g., hyperparameters used
        run.config.update({...})

        # Download the artifacts needed
        artifact = run.use_artifact(...)
        artifact_path = artifact.file()
        df = pd.read_parquet(artifact_path)

        # Do the WORK and log steps
        # The real component functionality goes HERE
        # ...
        # ...
        logger.info("Work done")

        # Upload any generated artifact(s)
        artifact = wandb.Artifact(...)
        artifact.add_file(...) # or .add_dir(...)
        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before any temp dir
        # gets deleted; this blocks the execution until then
        artifact.wait()

        # Log metrics, images, etc.
        run.log(...) # images, series of values (e.g., accuracy along time)
        run.summary['metric_A'] = ... # one value

if __name__ == "__main__":
    # Define and parse arguments
    parser = argparse.ArgumentParser(...)
    parser.add_argument(...)
    # ...
    args = parser.parse_args()

    # Execute component function
    go(args)

```

### Tips and Tricks

- `tempfile.TemporaryDirectory()`, `tempfile.NamedTemporaryFile()`,  or `os.remove(filename)`
- `wandb.init(project="my_project", ...)`
- `ml_pipeline.log`
- Ignore: `wandb`, `artifacts`, `outputs`, `mlruns`, `ml_pipeline.log`

## Improvements, Next Steps

- [ ] Fix the logging of the `check_data` component, which works with `pytest`.

## Interesting Links

- This repository doesn't focus on the techniques for data processing and modeling; if you are interested in those topics, you can visit my  [Guide on EDA, Data Cleaning and Feature Engineering](https://github.com/mxagar/eda_fe_summary).
- This project creates an inference pipeline managed with [MLflow](https://www.mlflow.org) and tracked with [Weights and Biases](https://wandb.ai/site); however, it is possible to define a production inference pipeline in a more simple way without the exposure to those 3rd party tools. In [this blog post](https://mikelsagardia.io/blog/machine-learning-production-level.html) I describe how to perform that transformation from research code to production-level code; the associated repository is [customer_churn_production](https://github.com/mxagar/customer_churn_production).
- If you are interested in more MLOps-related content, you can visit my notes on the [Udacity Machine Learning DevOps Engineering Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821): [mlops_udacity](https://github.com/mxagar/mlops_udacity).


## Authorship

Mikel Sagardia, 2022.  
No guarantees.

If you find this repository helpful and use it, please link to the original source: [https://github.com/mxagar/music_genre_classification](https://github.com/mxagar/music_genre_classification).