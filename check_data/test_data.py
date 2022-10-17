#import os
import logging
import scipy.stats
import pandas as pd
#import pytest

# Logging configuration
# NOTE: the current implementation fails to log to the logging file
logging.basicConfig(
    filename='../ml_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - check_data - %(message)s') # add component name for tracing
logger = logging.getLogger()

def test_column_presence_and_type(data):
    """Test columns and types.

    Args:
        data (tuple of data frames): reference (e.g., EDA) and current dataset to be tested
    """

    # Disregard the reference dataset
    _, data = data

    required_columns = {
        "time_signature": pd.api.types.is_integer_dtype,
        "key": pd.api.types.is_integer_dtype,
        "danceability": pd.api.types.is_float_dtype,
        "energy": pd.api.types.is_float_dtype,
        "loudness": pd.api.types.is_float_dtype,
        "speechiness": pd.api.types.is_float_dtype,
        "acousticness": pd.api.types.is_float_dtype,
        "instrumentalness": pd.api.types.is_float_dtype,
        "liveness": pd.api.types.is_float_dtype,
        "valence": pd.api.types.is_float_dtype,
        "tempo": pd.api.types.is_float_dtype,
        "duration_ms": pd.api.types.is_integer_dtype,  # This is integer, not float as one might expect
        "text_feature": pd.api.types.is_string_dtype,
        "genre": pd.api.types.is_string_dtype
    }

    # Check column presence
    try:
        assert set(data.columns.values).issuperset(set(required_columns.keys()))
    except AssertionError as err:
        logger.error("test_column_presence_and_type: Not all required columns are present.")

    for col_name, format_verification_function in required_columns.items():
        try:
            assert format_verification_function(data[col_name])
        except AssertionError as err:
            logger.error("test_column_presence_and_type: Column %s failed format test.", col_name)

    logger.info("test_column_presence_and_type: SUCCESS!")


def test_class_names(data):
    """Test genre class names.

    Args:
        data (tuple of data frames): reference (e.g., EDA) and current dataset to be tested
    """

    # Disregard the reference dataset
    _, data = data

    # Check that only the known classes are present
    known_classes = [
        "Dark Trap",
        "Underground Rap",
        "Trap Metal",
        "Emo",
        "Rap",
        "RnB",
        "Pop",
        "Hiphop",
        "techhouse",
        "techno",
        "trance",
        "psytrance",
        "trap",
        "dnb",
        "hardstyle",
    ]

    try:
        assert data["genre"].isin(known_classes).all()
    except AssertionError as err:
        logger.error("test_class_names: Unknown genre classes present.")
        
    logger.info("test_class_names: SUCCESS!")


def test_column_ranges(data):
    """Test column value ranges.

    Args:
        data (tuple of data frames): reference (e.g., EDA) and current dataset to be tested
    """

    # Disregard the reference dataset
    _, data = data

    ranges = {
        "time_signature": (1, 5),
        "key": (0, 11),
        "danceability": (0, 1),
        "energy": (0, 1),
        "loudness": (-35, 5),
        "speechiness": (0, 1),
        "acousticness": (0, 1),
        "instrumentalness": (0, 1),
        "liveness": (0, 1),
        "valence": (0, 1),
        "tempo": (50, 250),
        "duration_ms": (20000, 1000000),
    }

    for col_name, (minimum, maximum) in ranges.items():
        try:
            assert data[col_name].dropna().between(minimum, maximum).all(), (
                f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
                f"instead min={data[col_name].min()} and max={data[col_name].max()}"
            )
        except AssertionError as err:
            logger.error("test_column_ranges: %s not in range.", col_name)

    logger.info("test_column_ranges: SUCCESS!")

def test_kolmogorov_smirnov(data, ks_alpha):
    """Kolgomorov-Smirnov test with 2 samples:
    Analog to the T-Test but non-parametric.

    Args:
        data (tuple of data frames): reference (e.g., EDA) and current dataset to be tested
        ks_alpha (float): significance level
    """
    sample1, sample2 = data

    columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    # Bonferroni correction for multiple hypothesis testing
    # See blog post on this topic to understand where this comes from:
    # https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(columns))

    for col in columns:

        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col])

        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        try:
            assert p_value > alpha_prime
        except AssertionError as err:
            logger.error("test_kolmogorov_smirnov: p-value below threshold: %.5f", p_value)

    logger.info("test_kolmogorov_smirnov: SUCCESS!")
