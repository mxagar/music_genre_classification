# Reproducible ML Pipelines: Data Cleaning / Preprocessing

```bash
# General call
# arguments need to be replaced by real ones
python run.py --input_artifact {input_artifact} \
                        --artifact_name {artifact_name} \
                        --artifact_type {artifact_type} \
                        --artifact_description {artifact_description}

# Call with real arguments
# from config.yaml
# BUT: make sure environment is correct!
# ... or use mlflow, below
python get_data.py --input_artifact {file_url} \
                    --artifact_name {artifact_name} \
                    --artifact_type {artifact_type} \
                    --artifact_description {artifact_description}
```

```bash
mlflow run . \
-P input_artifact=... \
-P artifact_name=... \
-P artifact_type=... \
-P artifact_description=...
```
