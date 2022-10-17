# Reproducible ML Pipelines: Data Checks

Data checks can be:

- Deterministic: column names, types, sizes, ranges, etc.
- Non-deterministic: statistical, e.g., T-Tests (parametric) / Kolgomorov-Smirnov Tests (non-parametric), etc.

The current implementation uses [pytest](https://docs.pytest.org/en/7.1.x/) to perform the checks. We need these files:

- `contest.py` where the `pytest` fixtures are defined.
- `test_*.py` where the actual tests are define.

We could perform manually data checks with this command:

```bash
# General call
# arguments values need to be replaced by real ones
# Any file which starts with test_ is executed
pytest . -vv --reference_artifact project/reference_artifact.csv:latest \
             --sample_artifact project/sample_artifact.csv:latest \
             --ks_alpha 0.05
```

However, note that

- we need to replace the artifact/argument values with their definitions in the `config.yaml` file,
- any used artifact needs to be created beforehand,
- and we need to be running in the correct environment, defined in `conda.yaml`

Also, we can run the tests with `mlflow`, which sets the environment:

```bash
# General call
# argument values need to be replaced by real ones
mlflow run . \
-P reference_artifact=project/reference_artifact.csv:latest \
-P sample_artifact=project/sample_artifact.csv:latest \
-P ks_alpha=0.05

# Call with real arguments
# as defined in ../config.yaml
mlflow run . \
-P reference_artifact=project/reference_artifact.csv:latest \
-P sample_artifact=project/sample_artifact.csv:latest \
-P ks_alpha=0.05
```