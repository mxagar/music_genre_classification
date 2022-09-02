import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        #assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            }
        )

    if "preprocess" in steps_to_execute:

        ## YOUR CODE HERE: call the preprocess step
        # --input_artifact {input_artifact}
        # --artifact_name {artifact_name}
        # --artifact_type {artifact_type}
        # --artifact_description {artifact_description}
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                # the project path is not necessary
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name": "preprocessed_data.csv",
                "artifact_type": "preprocessed_data",
                "artifact_description": "Preprocessed data: missing values imputed and duplicated dropped"
            }
        )
        
    if "check_data" in steps_to_execute:

        ## YOUR CODE HERE: call the check_data step
        # --reference_artifact {reference_artifact}
        # --sample_artifact {sample_artifact}
        # --ks_alpha {ks_alpha}
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": "preprocessed_data.csv:latest",
                "ks_alpha": config["data"]["ks_alpha"]
            }
        )

    if "segregate" in steps_to_execute:

        ## YOUR CODE HERE: call the segregate step
        # --input_artifact {input_artifact}
        # --artifact_root {artifact_root}
        # --artifact_type {artifact_type}
        # --test_size {test_size}
        # --random_state {random_state}
        # --stratify {stratify}
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact": "preprocessed_data.csv:latest",
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"],
                #"random_state": 42, # already default value in MLproject
                "stratify": config["data"]["stratify"]
            }
        )
        
    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        ## YOUR CODE HERE: call the random_forest step
        # --train_data {train_data}
        # --model_config {model_config}
        # --export_artifact {export_artifact}
        # --random_seed {random_seed}
        # --val_size {val_size}
        # --stratify {stratify}
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": "data_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["random_forest_pipeline"]["random_forest"]["random_state"],
                "val_size": config["data"]["val_size"],
                "stratify": config["data"]["stratify"]
            }
        )

    if "evaluate" in steps_to_execute:

        ## YOUR CODE HERE: call the evaluate step
        # --model_export {model_export}
        # --test_data {test_data}
        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data": "data_test.csv:latest"
            }
        )

if __name__ == "__main__":
    go()
