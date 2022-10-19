#!/usr/bin/env python
import sys
import os
import logging
import argparse
import itertools

import yaml
import tempfile
import mlflow
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import wandb
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

# Add root path so that utilities package is found (for custom transformations)
sys.path.insert(1, '..')
from util_lib import ModeImputer

# Logging configuration
logging.basicConfig(
    filename='../ml_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - train - %(message)s') # add component name for tracing
logger = logging.getLogger()

def go(args):

    # Set default project name
    # but allow to override it via config.yaml with main.py or CLI with hydra
    project_name = "music_genre_classification"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.environ["WANDB_PROJECT"]    
    run = wandb.init(project=project_name, job_type="train")

    logger.info("Downloading and reading train artifact %s.", args.train_data)
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe.")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val.")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=df[args.stratify] if args.stratify != "null" else None,
        random_state=args.random_seed,
    )

    logger.info("Setting up pipeline.")
    pipe, used_columns = get_training_inference_pipeline(args)

    logger.info("Fitting.")
    pipe.fit(X_train[used_columns], y_train)

    # Evaluate / Validation
    pred = pipe.predict(X_val[used_columns])
    pred_proba = pipe.predict_proba(X_val[used_columns])

    logger.info("Scoring ROC-AUC.")
    score = roc_auc_score(y_val, pred_proba, average="macro", multi_class="ovo")

    run.summary["AUC"] = score

    # Export if required
    if args.export_artifact != "null":
        logger.info("Uploading inference artifact.")
        export_model(run, pipe, used_columns, X_val, pred, args.export_artifact)

    # Some useful plots
    logger.info("Plotting: feature importances, confusion matrix.")    
    fig_feat_imp = plot_feature_importance(pipe)

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_val[used_columns],
        y_val,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm),
        }
    )


def get_training_inference_pipeline(args):
    """Generate ML pipeline, 
    which consists of feature processing
    and a classifier (random forest).
    The training is not performed here.
    The processing step will impute missing values, encode the labels,
    normalize numerical features and compute a TF-IDF for the textual
    feature.

    Args:
        args (named tuple, CLI arguments): parsed CLI arguments

    Returns:
        pipe (sklearn Pipeline): processing + classification pipeline, untrained
        used_columns (list): list of used columns

    """
    # Get the configuration for the pipeline
    # passed as a YAML filename.
    # Note that
    # 1. if we run this script from the root level the YAML file is generated on-the-fly
    # by omegaconf using ../config.yaml and destroyed afterwards.
    # 2. if we run this locally for testing, we use the local random_forest_pipeline.yaml
    # but we should not do that in production! Instead, we should use the root level run! 
    with open(args.model_config) as fp: # random_forest_pipeline.yaml
        model_config = yaml.safe_load(fp)
    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)

    # We need 3 separate processing "tracks":
    # - one for categorical features
    # - one for numerical features
    # - one for textual ("nlp") features
    # Each of the tracks should be configured as a pipeline
    # created by make_pipeline().
    # Transformations are defined as sklearn.preprocessing classes.
    # If we want special/custom transformations, we need to define them
    # as done in transformations.py
    
    # Categorical processing pipeline
    # Here, I show how a custom transformer is used
    categorical_features = sorted(model_config["features"]["categorical"])
    categorical_transformer = make_pipeline(
        #SimpleImputer(strategy="constant", fill_value=0), OrdinalEncoder()
        ModeImputer(variables=categorical_features), OrdinalEncoder()
    )
    
    # Numerical processing pipeline
    numeric_features = sorted(model_config["features"]["numerical"])
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    
    # Textual ("nlp") processing pipeline
    nlp_features = sorted(model_config["features"]["nlp"])
    # This trick is needed because SimpleImputer wants a 2d input, but
    # TfidfVectorizer wants a 1d input. So we reshape in between the two steps
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    nlp_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=True, max_features=model_config["tfidf"]["max_features"]
        ),
    )
    
    # Put the 3 tracks together into one pipeline using the ColumnTransformer
    # This also drops the columns that we are not explicitly transforming
    processor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("nlp1", nlp_transformer, nlp_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Get a list of the columns we used
    used_columns = list(itertools.chain.from_iterable([x[2] for x in processor.transformers]))

    # Append classifier to processing pipeline.
    # Now we have a full prediction pipeline.
    # Pipeline needs to be used here.
    # The result is a pipeline with two high-level elements: processor and classifier.
    # Note that we pass the configuration dictionary to the model.
    pipe = Pipeline(
        steps=[
            ("processor", processor),
            ("classifier", RandomForestClassifier(**model_config["random_forest"])),
        ]
    )
    
    return pipe, used_columns


def plot_feature_importance(pipe):
    """Generate plot of feature importances.

    Args:
        pipe (object, pipeline): ML pipeline
            consisting of processsing + classifier

    Returns:
        fig_feat_imp (object, matplotlib figure): bar plot of feature importances
    """
    # We collect the feature importance for all non-nlp features first
    # Recall that pipe["processor"] is a ColumnTransformer,
    # which contains a list of parallel transformations in transformers
    feat_names = np.array(
        pipe["processor"].transformers[0][-1] # numeric_features
        + pipe["processor"].transformers[1][-1] # categorical_features
    )
    
    # Get feature importances of the classifier
    # but of numerical and categorical features
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]
    
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    # If we had one-hot encoding, we could consider doing something similar
    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names) :])
    
    # Features: importance values and names
    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "title + song_name")
    
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1] # inverse order
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)
    fig_feat_imp.tight_layout()
    
    return fig_feat_imp


def export_model(run, pipe, used_columns, X_val, val_pred, export_artifact):
    """Export (trained) inference pipeline,
    which consists of the processing
    and the classification model.

    Args:
        run (object): W&B run object
        pipe (object): sklearn pipeline
        used_columns (list): list of column/feature names
        X_val (object): features for the validation set
        val_pred (object): prediction for X_val
        export_artifact (str): name of the artifact to be exported
    """
    # Infer the signature of the model

    # Get the columns that we are really using from the pipeline
    signature = infer_signature(X_val[used_columns], val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        # The serialization mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        # is the default, better cross-system compatibility
        mlflow.sklearn.save_model(
            pipe, # our pipeline
            export_path, # path to a directory for the produced package
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature, # input and output schema
            input_example=X_val.iloc[:2] # the first few examples
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Random Forest pipeline export",
        )
        # We upload the complete directory!
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YAML file containing the configuration for the random forest",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random number generator.",
        required=False,
        default=42
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=False,
        default=0.3
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Name of a column to be used for stratified sampling. Default: 'null', i.e., no stratification",
        required=False,
        default="null",
    )

    args = parser.parse_args()

    go(args)
