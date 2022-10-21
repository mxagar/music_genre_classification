# Reproducible ML Pipelines: Data Checks (Step 3/6)

As introduced in the root `README.md`, we have the following generic steps in a reproducible ML pipeline:

1. Get data: `get_data`
2. Clean / Proprocess: `preprocess`
3. **Check data: `check_data`**
4. Segregate data: `segregate`
5. Inference Pipeline: Process + Train + Validate: `train_random_forest`
6. Evaluate: `evaluate`

![Generic Reproducible Pipeline](../assets/Reproducible_Pipeline.png)

This folder deals with the step or component **number 3: Check Data**.

This component is executed by the root level `mlflow` command, which gets the configuration parameters either **(1) from the root `config.yaml` using `hydra` (2) or these are hard-coded in the `main.py` script from the root level**. MLflow sets the required environment defined in the current/local `conda.yaml` automatically. We can also run this component locally and independently from the general project by invoking the local `MLproject` file as follows:

```bash
# The MLproject file in . is used
# Here, we use the same artifact for reference and sample
# In a real project, we would have 2 different!
mlflow run . \
-P reference_artifact="preprocessed_data.csv:latest" \
-P sample_artifact="preprocessed_data.csv:latest" \
-P ks_alpha=0.05
```

However, my recommended approach is to first run the entire pipeline once and then the selected components from the root level:

```bash
# Execute everything once at root level
mlflow run .
# Run selected components at root level
mlflow run . -P hydra_options="main.execute_steps='check_data'"
```

MLflow configures the necessary conda environment and executes the following command:

```bash
# General call, without argument values
# We execute python instead of pytest to enable logging;
# pytest is invoked in __main__
# pytest -s -vv . \
python test_data.py \
--reference_artifact {reference_artifact} \
--sample_artifact {sample_artifact} \
--ks_alpha {ks_alpha}

# Call with real arguments
# from ../config.yaml or ../main.py
# BUT: make sure environment is correct!
# ... or just use the mlflow call above :)
# We execute python instead of pytest to enable logging;
# pytest is invoked in __main__
# pytest -s -vv . \
python test_data.py \
--reference_artifact "preprocessed_data.csv:latest" \
--sample_artifact "preprocessed_data.csv:latest" \
--ks_alpha 0.05
```

Note that any artifact downloaded/used by this component needs to be already uploaded by previous components.

After executing `mlflow` we generate many outputs:

- The folders `mlruns`, `wandb` and `artifacts` are created or repopulated.
- We can see tracking information in the W&B web interface.

The current implementation uses [pytest](https://docs.pytest.org/en/7.1.x/) to perform the checks. We need these files:

- `contest.py` where the `pytest` fixtures are defined, along with the W&B `run`.
- `test_*.py` where the actual tests are define using pytest fixtures.

Therefore, the structure of the data validation/check component is different to the rest of the components.

Also, note that data checks can be:

- Deterministic: column names, types, sizes, ranges, etc.
- Non-deterministic: statistical, e.g., T-Tests (parametric) / Kolgomorov-Smirnov Tests (non-parametric), etc. To perform these types of tests we need at least a reference dataset in addition to the current one tested.

:warning: Important note: the current implementation fails to perform logging into the external logging file `../ml_pipeline.log`. I tried to [invoke pytest from python code](hhttps://docs.pytest.org/en/7.1.x/how-to/usage.html#calling-pytest-from-python-code) by defining `__main__`  in `test_data.py` but it didn't work.
