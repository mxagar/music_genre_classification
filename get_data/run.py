#!/usr/bin/env python
import os
import argparse
import logging
import pathlib
import wandb
import requests
import tempfile

# Logging configuration
logging.basicConfig(
    filename='../ml_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - get_data - %(message)s') # add component name for tracing
logger = logging.getLogger()

def go(args):

    # Derive the base name of the file from the URL
    basename = pathlib.Path(args.file_url).name.split("?")[0].split("#")[0]

    # Download file, streaming so we can download files larger than
    # the available memory. We use a named temporary file that gets
    # destroyed at the end of the context, so we don't leave anything
    # behind and the file gets removed even in case of errors
    logger.info("Downloading %s ...", args.file_url)
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:

        logger.info("Creating run")
        # Set default project name
        # but allow to override it via config.yaml with main.py or CLI with hydra
        project_name = "music_genre_classification"
        if "WANDB_PROJECT" in os.environ:
            project_name = os.environ["WANDB_PROJECT"]
        with wandb.init(project=project_name, job_type="get_data") as run:
            # Download the file streaming and write to open temp file
            with requests.get(args.file_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)

            # Make sure the file has been written to disk before uploading
            # to W&B
            fp.flush()

            logger.info("Creating artifact: %s", args.artifact_name)
            artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description,
                metadata={'original_url': args.file_url}
            )
            artifact.add_file(fp.name, name=basename)

            logger.info("Logging artifact: %s", basename)
            run.log_artifact(artifact)

            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
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
