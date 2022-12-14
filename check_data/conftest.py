import os
import pytest
import pandas as pd
import wandb

# Set default project name
# but allow to override it via config.yaml with main.py or CLI with hydra
project_name = "music_genre_classification"
if "WANDB_PROJECT" in os.environ:
    project_name = os.environ["WANDB_PROJECT"]
run = wandb.init(project=project_name, job_type="data_tests")

def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")

# Since we specify scope="session"
# this fixture is run once for the complete session.
# As an effect, the variable `data` triggers the function data() only once
# and it yields the loaded dataframe.
# Using `data` as an argument is equivalent to:
# data_ = data()
# test_data_length(data_)
# Note that with scope="function"
# every test_ function using data would load the complete dataset every time
@pytest.fixture(scope="session")
def data(request):

    reference_artifact = request.config.option.reference_artifact

    if reference_artifact is None:
        pytest.fail("--reference_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact

    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    local_path = run.use_artifact(reference_artifact).file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact(sample_artifact).file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2


@pytest.fixture(scope='session')
def ks_alpha(request):
    ks_alpha = request.config.option.ks_alpha

    if ks_alpha is None:
        pytest.fail("--ks_threshold missing on command line")

    return float(ks_alpha)
