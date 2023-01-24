# Music Genre Classification: A Boilerplate Reproducible ML Pipeline with MLflow and Weights & Biases

This project is a boilerplate for generating reproducible Machine Learning (ML) pipelines which are tracked and produce deployable inference artifacts. The used tools are:

- [MLflow](https://www.mlflow.org) for reproduction and management of pipeline processes.
- [Weights and Biases](https://wandb.ai/site) for artifact and execution tracking.
- [Hydra](https://hydra.cc) for configuration management.
- [Conda](https://docs.conda.io/en/latest/) for environment management.
- [Pandas](https://pandas.pydata.org) for data analysis.
- [Scikit-Learn](https://scikit-learn.org/stable/) for data modeling.
  
The pipeline is divided into the typical steps or components in an ML training pipeline, carried out in order. 

The example comes originally from an exercise in the repository [udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises), which I completed and extended with comments and some other minor features.

The used dataset is a modified version of the [songs in Spotify @ Kaggle](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify): in contains 40k+ song entries, each with 12 features, and the target variable is the genre they belong to. More information on the dataset can be found in the folder [`data_analysis`](data_analysis), which is not necessarily part of pipeline.

All the tracking done with Weights and Biases (W&B) can be accessed at the project page: [wandb.ai/datamix-ai/music_genre_classification](https://wandb.ai/datamix-ai/music_genre_classification).

Table of contents:

- [Music Genre Classification: A Boilerplate Reproducible ML Pipeline with MLflow and Weights \& Biases](#music-genre-classification-a-boilerplate-reproducible-ml-pipeline-with-mlflow-and-weights--biases)
  - [Overview of Boilerplate Project Structure](#overview-of-boilerplate-project-structure)
    - [Data Analysis and Serving](#data-analysis-and-serving)
  - [How to Use this Guide](#how-to-use-this-guide)
  - [Dependencies](#dependencies)
  - [How to Run This: Pipeline Creation and Deployment](#how-to-run-this-pipeline-creation-and-deployment)
    - [Run the Pipeline to Generate the Inference Artifact(s)](#run-the-pipeline-to-generate-the-inference-artifacts)
    - [Deployment: Use the Inference Artifacts for Performing Predictions](#deployment-use-the-inference-artifacts-for-performing-predictions)
  - [Notes Hydra, MLflow and Weights \& Biases](#notes-hydra-mlflow-and-weights--biases)
    - [Component Script Structure: `run.py`](#component-script-structure-runpy)
    - [Tracked Experiments and Hyperparameter Tuning](#tracked-experiments-and-hyperparameter-tuning)
      - [Hyperparameter Tuning with Hydra Sweeps](#hyperparameter-tuning-with-hydra-sweeps)
    - [MLflow Tracking and W\&B Model Registries](#mlflow-tracking-and-wb-model-registries)
    - [Tips and Tricks](#tips-and-tricks)
  - [Improvements, Next Steps](#improvements-next-steps)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## Overview of Boilerplate Project Structure

In general, we can say that an ML pipeline has two types of elements:

- **Artifacts**: objects of any kind which are generated or used along the pipeline, e.g., raw or processed datasets, serialized models, etc.
- **Components** or **steps**: sub-processes that need to be carried out in order to generate an inference pipeline that is able to score new data, e.g., fetch the dataset, clean it, etc.

In this boilerplate, Weights & Biases is used to track artifacts and component executions and MLflow is used to manage those component executions. The following figure summarizes how these ideas are implemented:

![Reproducible ML Pipeline: Generic Workflow](assets/Reproducible_Pipeline.png)

The final product of the ML pipeline is the **inference artifact**, which consists of the *processing pipeline* and the *trained model*.

Note that some artifacts and components have been marked with lighter colors and dashed lines:

- EDA: although the data analysis is necessary, it is considered prior to the generation of the ML pipeline and it should be stand-alone. However, it can be tracked and it uses the uploaded dataset, as shown in its dedicated folder `data_analysis`.
- Process: The data processing can be performed before or inside the training component; if it's carried out before, the processing pipeline must be exported, along with the processed dataset. If it's carried out in the training component, we can skip exporting those, since the processing becomes part of the inference pipeline. In general, having an extra processing component is not that uncommon in small projects; however, as projects gain in complexity, it is **very recommendable to integrate all the processing in the training component within a `Pipeline` object**. **For the rest of this project, I will assume that the processing is inside the training component.** In any case, new data should be transformed with any processing we define.

The file structure of the root folder reflects the sequence diagram of the previous figure:

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
│   └── README.md
├── preprocess/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
│   └── README.md
├── check_data/
│   ├── MLproject
│   ├── conda.yml
│   ├── conftest.py
│   └── test_data.py
│   └── README.md
├── segregate/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
│   └── README.md
├── train_random_forest/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
│   └── README.md
├── evaluate/
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
│   └── README.md
├── util_lib/
│   ├── __init__.py
│   ├── transformations.py
│   └── README.md
└── serving/
    ├── README.md
    ├── serving_example.py
    └── ...
```

The most important high-level files are `config.yaml` and `main.py`; they contain the parameters and the main pipeline execution order, respectively. Each component or pipeline step has its own project sub-folder, with their `MLproject` and `conda.yaml` files, for `mlflow` and conda environment configuration, respectively. If you'd like to know how these files interact with each other, please look at the section [Notes Hydra, MLflow and Weights & Biases](#notes-hydra-mlflow-and-weights--biases).

As for the pipeline steps or components, each one has a dedicated folder with a `README.md`; here's a summary of what happens in each component:

1. [`get_data/`](get_data)
    - A parquet file of songs and their attributes is downloaded from a URL; the songs need to be classified according to their genre.
    - The dataset it uploaded to Weights and Biases as an artifact.
2. [`preprocess/`](preprocess)
    - Raw dataset artifact is downloaded and preprocessed.
    - The data preprocessing done here is very basic  and specific for the training dataset: cleaning, duplicate removal, etc.
    - Any transformation of feature engineering required also on new data points should not go here, but in the processing of the `train_random_forest` component.
3. [`check_data/`](check_data)
    - Data validation: pre-processed dataset is checked using `pytest`.
    - In the dummy example, the reference and sample datasets are the same, and only deterministic tests are carried out, but we could have used another reference dataset for non-deterministic (i.e., statistical) tests.
4. [`segregate/`](segregate)
    - Train/test split is done and the two splits are uploaded as artifacts.
5. [`train_random_forest/`](random_forest)
    - Component/step with which a random forest model is defined and trained.
    - The training split is subdivided to train/validation.
    - The model is packed in a pipeline which contains data preprocessing and the model itself
        - The data preprocessing differentiates between numerical /categorical / NLP columns with `ColumnTransformer` and performs, among others, imputations, re-shapings, text feature extraction (TDIDF) and scaling.
        - The initial model configuration is achieved via a temporary YAML file created in the `main.py` module using the parameters from `config.yaml`.
    - That inference pipeline is exported and uploaded as an artifact.
    - Performance metrics (AUC) and images (confusion matrix, feature importances) are generated and uploaded as artifacts.
6. [`evaluate/`](evaluate)
    - The artifacts related to the test split and the inference pipeline are downloaded and used to compute the metrics with the test dataset.

Obviously, not all steps need to be carried out every time; to that end, with have the parameter `main.execute_steps` in the `config.yaml`. We can override it when calling `mlflow run`, as shown in the section [How to Run This](#how-to-run-pipeline-creation-and-deployment).

Note that the folder [`util_lib`](util_lib) is a utility package used by some of the components; in particular, it contains custom feature transformers that are employed in `train_random_forest` and `evaluate`. It can be extended with further shared functionalities.

### Data Analysis and Serving

There are two stand-alone folders/steps that are not part of the training pipeline; each of them has an explanatory file that extends the current `README.md`:

- [`data_analysis`](data_analysis/README.md): simple Exploratory Data Analysis (EDA), Data Cleaning and Feature Engineering (FE) are performed, as well as data modeling with cross validation to find the optimum hyperparameters. This folder contains an exemplary research environment crafted prior to the pipeline and it serves as a sandbox for testing ideas.
- [`serving`](serving/README.md): the exported inference pipeline artifact is deployed to production using different approaches.

## How to Use this Guide

1. Optionally, have a look at [`data_analysis`](data_analysis/README.md): a simple data preprocessing (+ EDA) is performed to understand the dataset.
2. Check the pipeline managed with [MLflow](https://www.mlflow.org) and [Weights and Biases](https://wandb.ai/site): Follow the notes on the [Overview](#overview-of-boilerplate-project-structure).
3. Optionally, have a look at the section [Notes Hydra, MLflow and Weights & Biases](#notes-hydra-mlflow-and-weights--biases) to understand how the implemented tools interact with each other.
4. Run the tracked pipeline as explained in [How to Run: Pipeline Creation and Deployment](#how-to-run-pipeline-creation-and-deployment); make sure you have installed the [dependencies](#dependencies).

## Dependencies

You need to have, at least:

- [Conda](https://docs.conda.io/en/latest/): environment management.
- [MLflow](https://www.mlflow.org): management of step parametrized step executions.
- [Weights and Biases](https://wandb.ai/site): tracking of experiments, artifacts, metrics, etc.

The rest is handled by the component-specific `conda.yaml` files. You can follow this short recipe to set up the main environment from which everything is launched:

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

It is possible to automate all that with a `conda.yaml` and `conda env create`, but I wanted to leave all steps to make clear what's going on, since this is a boilerplate :wink:

## How to Run This: Pipeline Creation and Deployment

This section deals with the creation and deployment of the inference pipeline.

First, we need to run the entire pipeline (all steps) at least once (locally or remotely) to generate all the artifacts. For that, we need to be logged with our Weights and Biases account. After that, we can perform online/offline predictions with MLflow. The correct model version & co. needs to be checked on the W&B web interface.

### Run the Pipeline to Generate the Inference Artifact(s)

In order to **run the complete project code locally**:

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

We can also run **isolated project components locally**; for that, we need to `cd` to the selected component and execute `mlflow` with the corresponding arguments. The `README.md` of each component provides the necessary call. However, note that any component needs input artifacts created upstream to be executed successfully -- thus, these must have been already created and stored in the W&B servers. My recommended approach is to first run the entire pipeline once and then the selected components from the root level:

```bash
# Execute everything once at root level
mlflow run .
# Run selected components at root level
mlflow run . -P hydra_options="main.execute_steps='get_data'"
```

Finally, since the repository is public, anyone can **run the complete project code remotely**; additionally, we can also create Github project/repository releases and run them with their tag:

```bash
# Go to a new empty folder
cd new-empty-folder

# General command
mlflow run -v [commit hash or branch name] [URL of your Github repo]
# Concrete command
mlflow run -v 82f17d94e0800811e81f4d55c0442d3189ed0a63 git@github.com:mxagar/music_genre_classification.git

# If we create a Release on Github, we can run that specific release remotely, too
# Create a release 0.0.1 on Github: 
#   https://github.com/mxagar/music_genre_classification/releases
#   Draft a new release
#   Choose tag: write a Major.Minor.Patch version number
# Then, to run that release tag remotely
mlflow run https://github.com/mxagar/music_genre_classification.git \
-v 0.0.1

# Project name changed
# We point to the branch main; usually, we should point to a branch like master / stable, etc.
# Note: currently, we cannot put tag names in -v
mlflow run git@github.com:mxagar/music_genre_classification.git -v main -P hydra_options="main.project_name=remote_execution"
```

### Deployment: Use the Inference Artifacts for Performing Predictions

See [`serving/README.md`](serving/README.md) for more information on:

- How to download and use the inference pipeline to perform predictions within a python script.
- How to download and use the inference pipeline to perform batch predictions via CLI.
- How to serve the inference pipeline as a REST API.
- How to create a docker image that serves the REST API.

Before serving or deploying anything, we need to have run the entire pipeline at least once, as explained above, e.g., locally:

```bash
# In the root project directory, where config.yaml is
mlflow run .
```

## Notes Hydra, MLflow and Weights & Biases

In this section, I provide a high level explanation of how all the tools interact in the boilerplate; it is intended as a short summary/refresher for my future self. For deeper explanations you can check tutorial and documentation links listed in the section [Interesting Links](#interesting-links).

As a reminder, I use

- [Weights and Biases](https://wandb.ai/site), or `wandb`, for managing artifacts and tracking runs/experiments,
- [MLflow](https://mlflow.org), or `mlflow`, for managing component/step executions,
- [Hydra](https://hydra.cc), or `hydra`, for controlling the configurations for MLflow,
- [Conda](https://docs.conda.io/en/latest/), or `conda`, for environment management,
- and [Pandas](https://pandas.pydata.org) and [Scikit-Learn](https://scikit-learn.org/stable/) for data analysis and modeling, respectively.

In summary, one could say that `hydra` configures `mlflow`, while `mlflow` manages the execution of pipeline components/steps in isolated `conda` environments, and these components are tracked with `wandb`. At end of the day, the goal is to have an ML pipeline 

- which can be easily **reproduced or re-run**,
- from which executions and artifacts are **tracked**,
- and which produces an easily **deployable inference artifact**.

Regarding `wandb` and its tracking functionality, we have the following elements:

- Runs: the minimum unit which is tracked; each component/step creates a unique run when it's executed. Additionally, we can assign a `job_type` tag to a run. Runs have usually automatic names.
- Experiments or groups: collections of runs, e.g., for development / research / production, etc.
- Projects: collections of runs with the same goal, e.g., the project or application itself; usually, all the components are defined under the same project.
- Artifacts: any file/directory produced during a run; they are all versioned and uploaded if anything changes in their content.

The interaction with `wand` in the code happens in the component modules themselves, and the section [Component Script Structure](#component-script-structure-runpy) shows a schematic but well documented example of the most common interactions. But it doesn't end there: all the tracking and lineage information, and more is on the web interface [https://wandb.ai/](https://wandb.ai/). 

Regarding `mlflow`, we also have `run` objects, but these are treated as execution commands of component modules. These executions happen via an `MLproject` configuration file. Each component has an `MLproject` file, which contains

- A command that is to be executed; this command usually executes the component module, e.g., `run.py`.
- The reference of the `conda.yaml` file that defines the environment.
- A list of parameters that are passed to the executed command as CLI arguments.

MLflow can add components hierarchically to the project; usually, we have a root `Mlproject` linked to a project module `main.py`. This module invokes the components located in different folders and in the desired order. Note that this invocation consists in running `MLprojects`, which in turn execute a command related to the component module. Also, this root-level project module `main.py` reads a `config.yaml` via a `hydra` decorator, which is used by all the components.

Let's consider a project or ML pipeline with two generic components; the file structure could look as follows:

![Simplified project structure](./assets/MLflow_WandB_Structure.png)

Everything might sound complicated, but the interaction is implemented in the files very easily and the following figure summarizes how this happens:

![Simplified project files](./assets/MLflow_WandB_Files.png)

### Component Script Structure: `run.py`

In the following, I provide a schematic but extensively commented skeleton of a component module; it extends the `run.py` example from the previous image and it shows how the most important `wandb` tracking calls are done.

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
    # IMPORTANT: 
    # 1) Set project name
    # to share artifacts between different components,
    # but allow to override it via config.yaml with main.py or CLI with hydra
    # 2) Set also a meaningful job type related to the component:
    # get_data, preprocess_data, check_data, split_data, train, evaluate, etc.
    project_name = "my_project"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.environ["WANDB_PROJECT"]
    with wandb.init(project=project_name, job_type="job_of_module") as run:

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

### Tracked Experiments and Hyperparameter Tuning

> :warning: Even though experiment tracking and hyperparameter tuning is mainly part of the `train_random_forest` component, I explain it here, because [hydra sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) are used for grid searches. These can be launched with `mlflow` where the `config.yaml` loaded by hydra is located, i.e., in the project root in our case.

Machine learning pipelines often require many iterations and experiments to solve issues characteristic to data modeling, for instance:

- Unexpected columns, unexpected values, etc.
- Class imbalance.
- Hyperparameter optimization after the validation.

Due to that extremely iterative nature of ML environments, we should:

- Plan iterations in the development process.
- Be systematic: change one thing at the time and track both code, parameters and data.

Additionally, each experiment needs to be **reproducible**; to that end, we need to uses randomness seeds whenever randomness is applied.

We should commit to our repository the code before running an experiment; that way, the code state is stored with a hash id and if we click on the **info button** of the run in the W&B web interface (left menu bar), we'll get the git checkout command that downloads the precise code status that was used for the experiment! For example:

```
git checkout -b "crisp-armadillo-2" c48420c28324e7b1b52aa84523b514f9944a21a0
```

In that info panel, we can see also the configuration parameters we add to the run/experiment in the code.

```python
import wandb

# New run
run = wandb.init(...)

# Storing hyper parameters
# We store parameters in dictionaries which can be nested
run.config.update({
    "batch_size": 128,
    "weight_decay": 0.01,
    "augmentations": {
        "rot_angle": 45,
        "crop_size": 224
    }
})

# Log one value
run.summary['accuracy'] = 0.9

# Time varying metrics - last reported in table
for i in range(10):
    run.log(
        {
            "loss": 1.2 - i * 0.1
        }
    )

# Log multiple time-varying metrics
for i in range(10):
    run.log(
        {
            "recall": 0.8 + i * 0.01,
            "ROC": 0.1 + i**2 * 0.01
        }
    )

# Explicit x-axis
for i in range(10):
    run.log(
        {
            "precision": 0.8 + i * 0.01,
            "epoch": i
        }
    )

# Plots
fig = plt.plot(...)
run.log({
    "image": wandb.Image(fig)
})

```

#### Hyperparameter Tuning with Hydra Sweeps

Hyperparameter tuning is one of the typical examples in which we iteratively run different experiments. We can perform grid searches with [hydra sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/); to that end, first, we need to add the dependencies `hydra-joblib-launcher` and `hydra-core`.

Then, we can override the parameters in the `config.yaml` and perform grid searches as follows:

```bash
## Experiment 1
# We override max_depth
mlflow run . \
-P hydra_options="random_forest_pipeline.random_forest.max_depth=5"

## Experiment 2
# We override n_estimators
mlflow run . \
-P hydra_options="random_forest_pipeline.random_forest.n_estimators=10"

## Experiment 3
# Override max_depth with values 1,5,10
mlflow run . \
-P hydra_options="main.execute_steps='train_random_forest' random_forest_pipeline.random_forest.max_depth=1,5,10 -m"
# Override max_depth with values 1-10 increased by 2
mlflow run . \
-P hydra_options="main.execute_steps='train_random_forest' random_forest_pipeline.random_forest.max_depth=range(1,10,2) -m"

## Experiment 4
# Sweep multiple parameters
# Note we can use range()
# Additionally, only the step/component train_random_forest is run
mlflow run . \
-P hydra_options="hydra/launcher=joblib main.execute_steps='train_random_forest' random_forest_pipeline.random_forest.max_depth=range(10,50,10) random_forest_pipeline.tfidf.max_features=range(50,200,50) -m"
```

At the end, we check the runs of the project in the table view of W&B: we hide all columns except the metric we want to optimize (AUC) and the hyperparameters we have varied (e.g., `tfidf.max_features`, `random_forest.max_depth`). We might decide not to choose the model with the best AUC, but the one with an AUC value close to the best but less complex (smaller depth and less features).

Since the CLI is limited to 250 characters, we can also perform parameter sweeps via the `config.yaml`; in that case we would simple execute `mlflow run .`.

Examples: 

- [Sweep demo](https://wandb.ai/example-team/sweep-demo)
- Associated Github repository is [pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion).
- [W&B examples](https://github.com/wandb/examples).

Finally, see also my notes on the **alternative of applying `GridSearchCV`** in `train_random_forest`: [README.md](./train_random_forest/README.md).

### MLflow Tracking and W&B Model Registries

In this boilerplate, I use

- Weights and biases for managing artifacts and tracking runs/experiments
- MLflow for managing component/step executions
- and hydra for controlling the configurations for MLflow.

However, these tools can do much more; for instance:

- [MLflow can track](https://www.mlflow.org/docs/latest/tracking.html).
- [W&B has a model registry](https://wandb.ai/registry/model).
- [Hydra can be used for more complex hierarchical configurations by composition and override through config files and the command line](https://hydra.cc/docs/intro/). We can also perform Bayesian optimization of the hyperparameters without modifying the pipeline!

### Tips and Tricks

- Commit & push always before running. The reason is that we can track simultaneously code version performances and generated artifacts. Additionally, W&B offers the functionality of running a specific commit hash version!
- Sometimes a component creates new local objects that are uploaded to W&B. After the upload, we don't need them locally and it's a good practice to remove them, e.g., with `os.remove(filename)`. Another option is using `with tempfile.TemporaryDirectory()` or `with tempfile.NamedTemporaryFile()` during the creation.
- We need to put all the W&B runs under the same project so that they can share the same artifacts, since these are named on the W&B system after `<project>/<name>:<version>`: `wandb.init(project="my_project", ...)`. We can control the overall project name with Hydra and the `config.yaml`, too, by setting the environment variable `WANDB_PROJECT`, as done in `main.py`. However, if we run isolated components, that environment variable alone doesn't assign the project name.
- Any time we add an artifact, the name doesn't have the version; but when we get/download an artifact, we need to specify it with `<artifact_name>:<version>`. If we have not specified the project name anywhere, it needs to be `<project_name>/<artifact_name>:<version>`. Usually we use `version=latest`, but we should check the W&B web interface to select the desired one.
- Use logging, outputted to the file `ml_pipeline.log` in this boilerplate. Note the structure of the logs. Unfortunately, I didn't manage to output test logs with the current setup, though -- fix is coming whenever I have time.
- When executing the components, folders are generated which we should (git) ignore: `wandb`, `artifacts`, `outputs`, `mlruns`, etc. For instance, artifact object will be in `artifacts/` and they will be managed by the `wandb` API.
- When adding components in `main.py` these can be online/git repositories; we just provide the proper URL and that's it. However, we might need to specify the branch with the parameter `version` in the `mlflow.run()` call, because `mlflow` defaults to use `master`. An example of this is shown in my repository [ml_pipeline_rental_prices](https://github.com/mxagar/ml_pipeline_rental_prices).
- A `wandb.Artifact()` has a parameter `metadata` which can take as value a dictionary; we can save there a configuration dictionary, e.g., in the case of an inference artifact, the model configuration parameters.
- In the current project, hydra loads the `config.yaml` file and extracts the model configuration part to a temporary file/dictionary using [OmegaConf](https://omegaconf.readthedocs.io/en/2.2_branch/). However, we can also do that without the OmegaConf dependency; an example is given in my repository [ml_pipeline_rental_prices](https://github.com/mxagar/ml_pipeline_rental_prices).
- Use Github releases to freeze concrete versions and enable remote execution (see section [Run the Pipeline to Generate the Inference Artifact(s)](#run-the-pipeline-to-generate-the-inference-artifacts)).
- Add the tag `prod` to the artifacts that will be used in production on the web interface of W&B; then, in the API call, we use the artifact name `<artifact_name>:prod`.
- The transformer `sklearn.preprocessing.FunctionTransformer` can be used to convert any custom function into a transformer! That's very useful! There is an example with a library function in the component `train_rancom_forest`; additionally, I have another example in my repository [ml_pipeline_rental_prices](https://github.com/mxagar/ml_pipeline_rental_prices).


To remove all `mlflow` environments that are created:

```bash
# Get a list of all environments that contain mlflow in their name
conda info --envs | grep mlflow | cut -f1 -d" "
# Remove the environments from the above list
for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y;done
```

## Improvements, Next Steps

- [ ] Fix the logging of the `check_data` component, which works with `pytest`.
- [ ] Create a (manual) docker image which serves the model.
- [ ] Add `GridSearchCV` to the component `train_random_forest` as done in [`Modeling.ipynb`](./data_analysis/Modeling.ipynb).

## Interesting Links

- This repository doesn't focus on the techniques for data processing and modeling; if you are interested in those topics, you can visit my  [Guide on EDA, Data Cleaning and Feature Engineering](https://github.com/mxagar/eda_fe_summary).
- This project creates an inference pipeline managed with [MLflow](https://www.mlflow.org) and tracked with [Weights and Biases](https://wandb.ai/site); however, it is possible to define a production inference pipeline in a more simple way without the exposure to those 3rd party tools. In [this blog post](https://mikelsagardia.io/blog/machine-learning-production-level.html) I describe how to perform that transformation from research code to production-level code; the associated repository is [customer_churn_production](https://github.com/mxagar/customer_churn_production).
- If you are interested in the automated deployment of production-ready ML pipelines packaged in APIs, check my example repository [census_model_deployment_fastapi](https://github.com/mxagar/census_model_deployment_fastapi).
- [Weights & Biases Model Registry](https://docs.wandb.ai/guides/models).
- [Machine learning model serving for newbies with MLflow](https://towardsdatascience.com/machine-learning-model-serving-for-newbies-with-mlflow-76f9f0ac3cb2).
- Another example where a reproducible ML pipeline is created using the same tools: [Reproducible Machine Learning pipeline that predicts short-term rental prices in New York](https://github.com/mxagar/ml_pipeline_rental_prices).
- If you are interested in more MLOps-related content, you can visit my notes on the [Udacity Machine Learning DevOps Engineering Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821): [mlops_udacity](https://github.com/mxagar/mlops_udacity).
- [Weights and Biases tutorials](https://wandb.ai/site/tutorials).
- [Weights and Biases documentation](https://docs.wandb.ai/).


## Authorship

Mikel Sagardia, 2022.  
No guarantees.

If you find this repository helpful and use it, please link to the original source.