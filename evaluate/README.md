# Reproducible ML Pipelines: Train (Step 5/6)

As introduced in the root `README.md`, we have the following generic steps in a reproducible ML pipeline:

1. Get data: `get_data`
2. Clean / Proprocess: `preprocess`
3. Check data: `check_data`
4. Segregate data: `segregate`
5. Inference Pipeline: Process + Train + Validate: `train_random_forest`
6. **Evaluate: `evaluate`**

![Generic Reproducible Pipeline](../assets/Reproducible_Pipeline.png)

This folder deals with the step or component **number 6: Evaluate**.

This component is executed by the root level `mlflow` command, which gets the configuration parameters either **(1) from the root `../config.yaml` using `hydra` (2) or these are hard-coded in the `../main.py` script from the root level**. MLflow sets the required environment defined in the current/local `conda.yaml` automatically. We can also run this component locally and independently from the general project by invoking the local `MLproject`.

Local call:

```bash
# The MLproject file in . is used
mlflow run . \
-P model_export="model_export:latest" \
-P test_data="data_test.csv:latest"
```

MLflow configures the necessary conda environment and executes the following command:

```bash
# General call, without argument values
python run.py \
--model_export {model_export} \
--test_data {test_data}

# Call with real arguments
# from ../config.yaml or ../main.py
# BUT: make sure environment is correct!
# ... or just use the mlflow call above :)
python run.py \
--model_export "model_export:latest" \
--test_data "data_test.csv:latest"
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

The component `evaluate` downloads the test split artifact and the inference artifact and computes the ROC-AUC of the model a well as its confusion matrix. Those results are uploaded as artifacts of the run.

The inference artifact can be imported/exported with API call `mlflow.sklearn.save_model() / load_model()`; it is not a single file, but a directory which contains these files:

```
.
├── MLmodel                 # YAML which describes the MLflow model
├── conda.yaml              # Conda env configuration
├── input_example.json      # Inpunt example
└── model.pkl               # Serialized model pipeline
```

If we don't specify another folder, it will be downloaded to `artifacts`.

