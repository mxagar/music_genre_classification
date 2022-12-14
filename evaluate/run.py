#!/usr/bin/env python
import sys
import os
import argparse
import itertools
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, plot_confusion_matrix

# Add root path so that utilities package is found (for custom transformations)
sys.path.insert(1, '..')
from util_lib import ModeImputer

# Logging configuration
logging.basicConfig(
    filename='../ml_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - evaluate - %(message)s') # add component name for tracing
logger = logging.getLogger()

def go(args):

    # Set default project name
    # but allow to override it via config.yaml with main.py or CLI with hydra
    project_name = "music_genre_classification"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.environ["WANDB_PROJECT"]    
    run = wandb.init(project=project_name, job_type="evaluate")

    logger.info("Downloading and reading test artifact %s.", args.test_data)
    test_data_path = run.use_artifact(args.test_data).file()
    df = pd.read_csv(test_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe.")
    X_test = df.copy()
    y_test = X_test.pop("genre") # get column and drop from frame!

    logger.info("Downloading and reading the exported model %s.", args.model_export)
    # Since this artifact contains a directory
    # and not a single file, you will have to use .download() instead of .file()
    model_export_path = run.use_artifact(args.model_export).download()
    # Load pipeline
    pipe = mlflow.sklearn.load_model(model_export_path)

    # Get features/columns that have been used for creating the pipeline
    used_columns = list(itertools.chain.from_iterable([x[2] for x in pipe['processor'].transformers]))
    # Predict ONLY with allowed columns/features; if it was trained like that!
    pred_proba = pipe.predict_proba(X_test[used_columns])

    # Evaluation: ROC-AUC
    logger.info("Scoring: ROC-AUC.")
    score = roc_auc_score(y_test, pred_proba, average="macro", multi_class="ovo")
    
    run.summary["AUC"] = score

    # Evaluation: Confusion matrix
    logger.info("Computing confusion matrix.")
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_test[used_columns],
        y_test,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "confusion_matrix": wandb.Image(fig_cm)
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)
