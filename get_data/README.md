# Reproducible ML Pipelines: Get Data (Step 1/5)

As introduced in the root `README.md`, we have the following generic steps in a reproducible ML pipeline:

1. Get data: `get_data`
2. Clean / Proprocess: `preprocess`
3. Check data: `check_data`
4. Segregate data: `segregate`
5. Inference Pipeline: Process + Train + Validate: `train_random_forest`
6. Evaluate: `evaluate`

![Generic Reproducible Pipeline](../assets/Reproducible_Pipeline.png)

This folder deals with the step or component number X: **X**.

The component is executed by the root level `mlflow` command, which gets the configuration parameters from the root `config.yaml` using `hydra` and sets the required environment defined in the current/local `conda.yaml`. However, we can also run it locally invoking the local `MLproject` file:

```bash
# The MLproject file in . is used
mlflow run . \
-P file_url=... \
-P artifact_name=... \
-P artifact_type=... \
-P artifact_description=...
```

MLflow configures the necessary conda environment and executes the following command:

```bash
# General call, without argument values
python get_data.py --file_url {file_url} \
                    --artifact_name {artifact_name} \
                    --artifact_type {artifact_type} \
                    --artifact_description {artifact_description}

# Call with real arguments
# from config.yaml
# BUT: make sure environment is correct!
# ... or just use the mlflow call above :)
python get_data.py --file_url {file_url} \
                    --artifact_name {artifact_name} \
                    --artifact_type {artifact_type} \
                    --artifact_description {artifact_description}
```

The script `get_data.py`, as most of the other components or step, has the following structure:

```python
# Imports
# ...

# Logger
logging.basicConfig(...)
logger = logging.getLogger()

# Component function
def go(args):
    # Instantiate W&B run in a context
    # or, if preferred, without a context
    # run = wandb.init(...)
    with wandb.init(...) as run:

        # Do the WORK and log steps   
        # ...
        logger.info("Work done")

        # Upload any generated artifact
        artifact = wandb.Artifact(...)
        artifact.add_file(...)
        run.log_artifact(artifact)
        artifact.wait()

        # Log images, etc.
        run.summary[...] = ...
        run.log(...)

if __name__ == "__main__":
    # Define and parse arguments
    parser = argparse.ArgumentParser(...)
    parser.add_argument(...)
    # ...
    args = parser.parse_args()

    # Execute component function
    go(args)

```

In the concrete case of the step `get_data`, 
