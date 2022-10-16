# Data Analysis and Modeling

This sub-folder is a stand-alone sub-project in which the Exploratory Data Analysis (EDA) and basic modeling is done prior to transferring all the code into the production environment project.

There are 3 notebooks:

- [`EDA_Tracking.ipynb`](EDA_Tracking.ipynb): a basic EDA is performed and in the process we track the notebook execution with W&B.
- [`Modeling.ipynb`](Modeling.ipynb): basic data modeling which is transferred later on to the production environment in the components of the higher level.
- [`UnsupervisedLearning.ipynb`](UnsupervisedLearning.ipynb): basic unsupervised learning.

In order to run the notebooks, **first**, we need to upload the dataset to W&B as an artifact.

```bash
# Folder where the dataset is located
cd ../dataset/

# Upload the dataset
# Project name: music_genre_classification
wandb artifact put \
      --name music_genre_classification/genres_mod.parquet \
      --type raw_data \
      --description "A modified version of the songs dataset" genres_mod.parquet

# Folder where the data analysis is carried out
cd ../data_analysis

# We can download the dataset to artifacts/ folder
# using the CLI; however, the notebook `EDA_Tracking`
# shows how to do it with the python API.
# Note the syntax: project_name/dataset_name:version
wandb artifact get music_genre_classification/genres_mod.parquet:latest
```

Note that in the production components the dataset is downloaded from an URL and added as an artifact, so we skip this CLI steps.

**Second**, we run `mlflow`, which launches the project defined in `MLproject`; this project basically installs a conda environment defined in `conda.yaml` and starts the Jupyter Notebook IDE. Within that Jupyter server instance, we can create all the notebooks we want.

```bash
mlflow run .

# The first time the environment is installed and it takes time.
# A Jupyter Notebook server should be opened in our default browser.
# The next times, it's much faster.
```

The most interesting notebook in terms of how to track notebooks and artifacts using W&B is [`EDA_Tracking.ipynb`](EDA_Tracking.ipynb); the steps carried out in that notebook are:

- Start a new run `run = wandb.init(project="music_genre_classification", save_code=True)`; `save_code=True` makes possible to track the code execution.
- Download the dataset artifact and explore it briefly.
- Perform a simple EDA:
  - Run `pandas_profiling.ProfileReport()`.
  - Drop duplicates.
  - Impute missing song and tile values with `''`.
  - Create new text field which is the concatenation of the title and the song name.
- Finish the run: `run.finish()`.

We can check in out W&B account that the artifact and the run(s) are registered and tracked.

The other two the notebooks perform operations that are in part transferred to the production components, but they are not tracked; the tracking is shown only in the notebook [`EDA_Tracking.ipynb`](EDA_Tracking.ipynb).

### Links

- Original dataset: [Dataset of songs in Spotify](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify).
- Example EDA and data modeling with the dataset [Understanding + classifying genres using Spotify audio features](https://www.kaylinpavlik.com/classifying-songs-genres/).

