from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pytest

from tpx3awkward import Tpx3Config
from tpx3awkward.processing import decode_tpx3_binary, cluster_decoded_df, raw_as_numpy
from tpx3awkward.processing.corrections import timewalk_corr, estimate_energies

RAW_DATA_DIR = Path(__file__).parents[1] / "data/raw/"
PROC_DATA_DIR = Path(__file__).parents[1] / "data/processed/"
CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.fixture
def decoded_df():
    return decode_tpx3_binary(raw_as_numpy(RAW_DATA_DIR / "raw_test_data_01.tpx3"))


@pytest.fixture
def stable_cdf():
    cdf =  pd.read_parquet(PROC_DATA_DIR / "raw_test_data_01_cent.parquet")
    cdf.loc[cdf['xc'] >= 255.5, 'xc'] -= 2
    cdf.loc[cdf['yc'] >= 255.5, 'yc'] -= 2
    return cdf


@pytest.fixture
def config():
    with Path(CONFIG_DIR/"tpx3_configurations.yaml").open() as f:
        data = yaml.safe_load(f)
    return Tpx3Config.model_validate(data)


def test_cluster_decoded_df(decoded_df, stable_cdf, config):
    current_cdf = cluster_decoded_df(decoded_df, config.time_window, config.radius)

    pd.testing.assert_frame_equal(current_cdf, stable_cdf.drop(columns=["e_sum", "t_corr"]), atol=0.01)


def test_cluster_decoded_df_only_tcorr(decoded_df, stable_cdf, config):
    decoded_df["t_corr"] = timewalk_corr(decoded_df["t"].to_numpy(), decoded_df["ToT"].to_numpy(), config.timewalk_b, config.timewalk_c)
    current_cdf = cluster_decoded_df(decoded_df, config.time_window, config.radius)

    pd.testing.assert_frame_equal(current_cdf, stable_cdf.drop(columns="e_sum"), atol=0.01)


def test_cluster_decoded_df_only_e(decoded_df, stable_cdf, config):
    decoded_df["e"] = estimate_energies(
        decoded_df["x"].to_numpy(),
        decoded_df["y"].to_numpy(),
        decoded_df["ToT"].to_numpy(),
        config.energy_estimation_parameters,
    )
    current_cdf = cluster_decoded_df(decoded_df, config.time_window, config.radius)

    pd.testing.assert_frame_equal(current_cdf, stable_cdf.drop(columns="t_corr"), atol=0.01)


def test_cluster_decoded_df_e_and_tcorr(decoded_df, stable_cdf, config):
    decoded_df["t_corr"] = timewalk_corr(decoded_df["t"].to_numpy(), decoded_df["ToT"].to_numpy(), config.timewalk_b, config.timewalk_c)
    decoded_df["e"] = estimate_energies(
        decoded_df["x"].to_numpy(),
        decoded_df["y"].to_numpy(),
        decoded_df["ToT"].to_numpy(),
        config.energy_estimation_parameters,
    )
    current_cdf = cluster_decoded_df(decoded_df, config.time_window, config.radius)

    pd.testing.assert_frame_equal(current_cdf, stable_cdf, atol=0.01)

