#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd
import wandb

# Logging configuration
logging.basicConfig(
    filename='../results.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - preprocess - %(message)s') # add component name for tracing
logger = logging.getLogger()

def go(args):

    run = wandb.init(project="music_genre_classification", job_type="preprocess_data")

    logger.info("Downloading artifact: %s", args.input_artifact)
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_parquet(artifact_path)

    # Drop the duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    # A minimal feature engineering step: a new feature
    logger.info("Basic feature engineering")
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    # Temporary file
    #filename = args.artifact_name
    filename = "processed_data.csv"
    df.to_csv(filename)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact: %s", args.artifact_name)
    run.log_artifact(artifact)

    # Remove created temporary file
    # we could also use tempfile.NamedTemporaryFile()
    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
