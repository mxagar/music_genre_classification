# Reproducible ML Pipelines: Get Data (Step 1/6)

As introduced in the root `README.md`, we have the following generic steps in a reproducible ML pipeline:

1. **Get data: `get_data`**
2. Clean / Proprocess: `preprocess`
3. Check data: `check_data`
4. Segregate data: `segregate`
5. Inference Pipeline: Process + Train + Validate: `train_random_forest`
6. Evaluate: `evaluate`

![Generic Reproducible Pipeline](../assets/Reproducible_Pipeline.png)

This folder deals with the step or component **number 1: Get Data**.

This component is executed by the root level `mlflow` command, which gets the configuration parameters either **(1) from the root `config.yaml` using `hydra` (2) or these are hard-coded in the `main.py` script from the root level**. MLflow sets the required environment defined in the current/local `conda.yaml` automatically. We can also run this component locally and independently from the general project by invoking the local `MLproject` file as follows:

```bash
# The MLproject file in . is used
mlflow run . \
-P file_url="https://github.com/mxagar/music_genre_classification/blob/main/dataset/genres_mod.parquet?raw=true" \
-P artifact_name="raw_data.parquet" \
-P artifact_type="raw_data" \
-P artifact_description="Data as downloaded"
```

MLflow configures the necessary conda environment and executes the following command:

```bash
# General call, without argument values
python run.py \
--file_url {file_url} \
--artifact_name {artifact_name} \
--artifact_type {artifact_type} \
--artifact_description {artifact_description}

# Call with real arguments
# from ../config.yaml or ../main.py
# BUT: make sure environment is correct!
# ... or just use the mlflow call above :)
python run.py \
--file_url "https://github.com/mxagar/music_genre_classification/blob/main/dataset/genres_mod.parquet?raw=true" \
--artifact_name "raw_data.parquet" \
--artifact_type "raw_data" \
--artifact_description "Data as downloaded"
```

After executing `mlflow` and, through it, the script `run.py`, we generate many outputs:

- The folders `mlruns`, `wandb` and `artifacts` are created or repopulated.
- The log file `../ml_pipeline.log` is modified.
- We can see tracking information in the W&B web interface.

The script `run.py`, as in most of the other components or steps, has the following structure:

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

In the concrete case of the step `get_data`, a `parquet` file is downloaded from the given URL and saved as a dataset artifact in W&B. The functionality `tempfile.NamedTemporaryFile()` is used for that.

Note that `data_analysis` shows how to upload manually the dataset via the CLI; thus, the current step is redundant, or it could be change to upload from a local folder.
