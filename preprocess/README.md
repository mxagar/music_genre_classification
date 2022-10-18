# Reproducible ML Pipelines: Data Cleaning / Preprocessing (Step 2/6)

As introduced in the root `README.md`, we have the following generic steps in a reproducible ML pipeline:

1. Get data: `get_data`
2. **Clean / Proprocess: `preprocess`**
3. Check data: `check_data`
4. Segregate data: `segregate`
5. Inference Pipeline: Process + Train + Validate: `train_random_forest`
6. Evaluate: `evaluate`

![Generic Reproducible Pipeline](../assets/Reproducible_Pipeline.png)

This folder deals with the step or component **number 2: Clean / Preprocess**.

This component is executed by the root level `mlflow` command, which gets the configuration parameters either **(1) from the root `config.yaml` using `hydra` (2) or these are hard-coded in the `main.py` script from the root level**. MLflow sets the required environment defined in the current/local `conda.yaml` automatically. We can also run this component locally and independently from the general project by invoking the local `MLproject` file as follows:

```bash
# The MLproject file in . is used
mlflow run . \
-P input_artifact="raw_data.parquet:latest" \
-P artifact_name="preprocessed_data.csv" \
-P artifact_type="preprocessed_data" \
-P artifact_description="Preprocessed data: missing values imputed and duplicated dropped"
```

MLflow configures the necessary conda environment and executes the following command:

```bash
# General call, without argument values
python run.py \
--input_artifact {file_url} \
--artifact_name {artifact_name} \
--artifact_type {artifact_type} \
--artifact_description {artifact_description}

# Call with real arguments
# from ../config.yaml or ../main.py
# BUT: make sure environment is correct!
# ... or just use the mlflow call above :)
python run.py \
--file_url "raw_data.parquet:latest" \
--artifact_name "preprocessed_data.csv" \
--artifact_type "preprocessed_data" \
--artifact_description "Preprocessed data: missing values imputed and duplicated dropped"
```

Note that any artifact downloaded/used by this component needs to be already uploaded by previous components.

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

In the concrete case of the step `preprocess`, a very basic data cleaning and feature engineering is performed. **Note that this preprocessing should be very basic, only to set the basic dataset; if any transformation should be applied to new incoming data, it should not go here, but in the inference pipeline, i.e., in `train_random_forest`.**

The preprocessed dataset is uploaded as a new artifact: `processed_data.csv`.

The preprocessing is the following:

```python
# Drop the duplicates
df = df.drop_duplicates().reset_index(drop=True)

# A minimal feature engineering step: a new feature
df['title'].fillna(value='', inplace=True)
df['song_name'].fillna(value='', inplace=True)
df['text_feature'] = df['title'] + ' ' + df['song_name']
```
