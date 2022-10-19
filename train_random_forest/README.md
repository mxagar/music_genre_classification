# Reproducible ML Pipelines: Train (Step 5/6)

As introduced in the root `README.md`, we have the following generic steps in a reproducible ML pipeline:

1. Get data: `get_data`
2. Clean / Proprocess: `preprocess`
3. Check data: `check_data`
4. Segregate data: `segregate`
5. **Inference Pipeline: Process + Train + Validate: `train_random_forest`**
6. Evaluate: `evaluate`

![Generic Reproducible Pipeline](../assets/Reproducible_Pipeline.png)

This folder deals with the step or component **number 5: Process, Train, Validate**.

This component is executed by the root level `mlflow` command, which gets the configuration parameters either **(1) from the root `../config.yaml` using `hydra` (2) or these are hard-coded in the `../main.py` script from the root level**. MLflow sets the required environment defined in the current/local `conda.yaml` automatically. We can also run this component locally and independently from the general project by invoking the local `MLproject`.

> :warning: This component should be called by the general `mlflow` command from the root level, because the random forest configuration parameters are extracted from `../config.yaml` by the `../main.py` script. However, I created a dummy `random_forest_pipeline.yaml` configuration file in the component folder for learning purposes; thus, we can run the component locally to see that it works. When the upper-level call is carried out, the `random_forest_pipeline.yaml` is ignored and the parameters are taken from `../config.yaml`.

Local call (to be avoided):

```bash
# The MLproject file in . is used
mlflow run . \
-P train_data="data_train.csv:latest" \
-P model_config="random_forest_pipeline.yaml" \
-P export_artifact="model_export" \
-P random_seed=42 \
-P val_size=0.3 \
-P stratify="genre"
```

MLflow configures the necessary conda environment and executes the following command:

```bash
# General call, without argument values
python run.py \
--train_data {train_data} \
--model_config {model_config} \
--export_artifact {export_artifact} \
--random_seed {random_seed} \
--val_size {val_size} \
--stratify {stratify}

# Call with real arguments
# from ../config.yaml or ../main.py
# BUT: make sure environment is correct!
# ... or just use the mlflow call above :)
python run.py \
--train_data "data_train.csv:latest" \
--model_config "random_forest_pipeline.yaml" \
--export_artifact "model_export" \
--random_seed 42 \
--val_size 0.3 \
--stratify "genre"
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

The component `train_random_forest` is very important, since it creates the inference artifact. **All the transformations required on new data points need to be defined here as processing steps in the inference pipeline.**

The main functionality function `go()` calls the following functions:

- `get_training_inference_pipeline()`: it generates the processing and classification pipeline.
  - A `Pipeline` is returned.
  - `ColumnTransformer()` is used to separate numerical, categorical and text columns; to each chunk, we assign a processing/transformation sub-`Pipeline`, as explained in the following.
  - `make_pipeline()` is used to create processing sub-`Pipelines` for each of the separated column chunks. We use `sklearn` or custom transformers.
  - All separated column chunks are reunited and a random forest classifier is appended.
  - Configuration parameters are obtained from the YAML.
- `plot_feature_importance()`: feature importances are plotted and uploaded to W&B.
- `export_model()`:
  - `mlflow.sklearn.save_model()` is used to serialize the model to a `tempfile.TemporaryDirectory()`.
  - The artifact is uploaded to W& (and then removed, since we create a temporal directory).

Note that the train split is further segregated in train/validation. The final training is done with the last train split and the generated plots/metrics are related to the validation split. 

Additional comments on the exported pipeline:

- The `sklearn` `Pipeline` is not the unique option for creating a pipeline. For Pytorch, we can use `torch.jit.script` instead; examples are provided in my notes on MLOps: [mlops_udacity](https://github.com/mxagar/mlops_udacity/blob/main/02_Reproducible_Pipelines/MLOpsND_ReproduciblePipelines.md).
- MLflow has several framework pipeline export/import functions in its API:
  - `mlflow.sklearn.save_model() / load_model()`
  - `mlflow.pytorch.save_model() / load_model()`
  - `mlflow.keras.save_model() / load_model()`
  - `mlflow.onnx.save_model() / load_model()` 
- When we export a pipeline, we can add two important elements to it:
  - a signature, which contains a the input/output schema
  - and input examples for testing
- MLflow figures out the correct conda environment automatically and generates the `conda.yaml` file. However, we can also explicitly override it with the `conda_env` option.
- The exported model can be converted to a Docker image that provides a REST API for the model.

The exported pipeline can be downloaded as artifact any time (see step `evaluate`). It consists of a directory with the following files:

```
.
├── MLmodel                 # YAML which describes the MLflow model
├── conda.yaml              # Conda env configuration
├── input_example.json      # Inpunt example
└── model.pkl               # Serialized model pipeline
```

## Tracked Experiments and Hyperparameter Tuning

See [`../README.md`](../README.md) for more information on how to perform iterative experiments and hyperparameter tuning using [hydra sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/).
