import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    # Any project name set in downstream components will be overwritten
    # We can change these variable names via CLI using hydra
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

    if "get_data" in steps_to_execute:
        # mlflow run . -P hydra_options="main.execute_steps='get_data'"
        _ = mlflow.run(
            os.path.join(root_path, "get_data"), # path of component
            "main", # entry point
            parameters={ # parameters passed to MLproject
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            }
        )

    if "preprocess" in steps_to_execute:
        # mlflow run . -P hydra_options="main.execute_steps='preprocess'"
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
        # mlflow run . -P hydra_options="main.execute_steps='check_data'"
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
        # mlflow run . -P hydra_options="main.execute_steps='segregate'"
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
        
    if "train_random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        # Whenever we have model or other object with many parameters
        # we should write config files for them.
        # That is easier than passing parameters in the code or via CLI
        # and we can guarantee compatibility in the code in case the model API changes
        # (i.e., we would simply change the config file).
        # Here a yaml is created from the relevant section of the config file
        # and passed as a dictionary to the model later on.
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        # mlflow run . -P hydra_options="main.execute_steps='train_random_forest'"
        _ = mlflow.run(
            os.path.join(root_path, "train_random_forest"),
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
        # mlflow run . -P hydra_options="main.execute_steps='evaluate'"
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
