from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tpx3awkward import (
    Tpx3Config,
    convert_tpx3_binary,
    convert_tpx3_file,
    convert_tpx3_files,
    convert_tpx3_files_parallel,
    read_parquet_config,
)
from tpx3awkward.processing import raw_as_numpy

RAW_DATA_DIR = Path(__file__).parents[1] / "data/raw/"
PROC_DATA_DIR = Path(__file__).parents[1] / "data/processed/"
CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_convert_tpx3_binary():
    path_to_data = RAW_DATA_DIR / "raw_test_data_01.tpx3"
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    energy_estimation_test_parameters = np.load(path_to_energy_parameters)
    cdf, tdc_df = convert_tpx3_binary(
        raw_as_numpy(path_to_data),
        estimate_energy=True,
        energy_estimation_parameters=energy_estimation_test_parameters,
        correct_timewalk=True,
        timewalk_b=167.0,
        timewalk_c=-0.016,
        verbose=True,
    )

    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
    assert required.issubset(cdf.columns)
    pd.testing.assert_frame_equal(cdf, pd.read_parquet(PROC_DATA_DIR / "raw_test_data_01_cent.parquet"), atol=0.01)

    # no tdc in this test data, so tdc_df should be None
    assert tdc_df is None


def test_convert_tpx3_binary_tdc():
    path_to_data = RAW_DATA_DIR / "tdc/raw_test_data_tdc_00.tpx3"
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    energy_estimation_test_parameters = np.load(path_to_energy_parameters)
    cdf, tdc_df = convert_tpx3_binary(
        raw_as_numpy(path_to_data),
        estimate_energy=True,
        energy_estimation_parameters=energy_estimation_test_parameters,
        correct_timewalk=True,
        timewalk_b=167.0,
        timewalk_c=-0.016,
        tdc=True,
        verbose=True,
    )

    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
    assert required.issubset(cdf.columns)

    required_tdc = {"tdc_t_ns", "tdc_type", "tdc_chip"}
    assert required_tdc.issubset(tdc_df.columns)


def test_convert_tpx3_binary_zero_bytes():
    path_to_data = RAW_DATA_DIR / "misc/zero_bytes.tpx3"
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    energy_estimation_test_parameters = np.load(path_to_energy_parameters)
    cdf, tdc_df = convert_tpx3_binary(
        raw_as_numpy(path_to_data),
        estimate_energy=True,
        energy_estimation_parameters=energy_estimation_test_parameters,
        correct_timewalk=True,
        timewalk_b=167.0,
        timewalk_c=-0.016,
        tdc=True,
        verbose=True,
    )

    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
    assert required.issubset(cdf.columns)

    required_tdc = {"tdc_t_ns", "tdc_type", "tdc_chip"}
    assert required_tdc.issubset(tdc_df.columns)


def test_convert_tpx3_file(tmp_path):
    path_to_data = RAW_DATA_DIR / "raw_test_data_01.tpx3"
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    energy_estimation_test_parameters = np.load(path_to_energy_parameters)
    convert_tpx3_file(
        path_to_data,
        output_dir=tmp_path,
        estimate_energy=True,
        energy_estimation_parameters=energy_estimation_test_parameters,
        correct_timewalk=True,
        timewalk_b=167.0,
        timewalk_c=-0.016,
        verbose=True,
    )

    cdf = pd.read_parquet(tmp_path / "raw_test_data_01_cent.parquet")
    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
    assert required.issubset(cdf.columns)
    pd.testing.assert_frame_equal(cdf, pd.read_parquet(PROC_DATA_DIR / "raw_test_data_01_cent.parquet"), atol=0.01)


def test_convert_tpx3_file_tdc(tmp_path):
    path_to_data = RAW_DATA_DIR / "tdc/raw_test_data_tdc_00.tpx3"
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    energy_estimation_test_parameters = np.load(path_to_energy_parameters)
    convert_tpx3_file(
        path_to_data,
        output_dir=tmp_path,
        estimate_energy=True,
        energy_estimation_parameters=energy_estimation_test_parameters,
        correct_timewalk=True,
        timewalk_b=167.0,
        timewalk_c=-0.016,
        tdc=True,
        verbose=True,
    )

    cdf = pd.read_parquet(tmp_path / "raw_test_data_tdc_00_cent.parquet")
    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
    assert required.issubset(cdf.columns)

    tdc_df = pd.read_parquet(tmp_path / "raw_test_data_tdc_00_tdc.parquet")
    required_tdc = {"tdc_t_ns", "tdc_type", "tdc_chip"}
    assert required_tdc.issubset(tdc_df.columns)

    pd.testing.assert_frame_equal(tdc_df, pd.read_parquet(PROC_DATA_DIR / "tdc/raw_test_data_tdc_00_tdc.parquet"), atol=0.01)


def test_convert_tpx3_file_no_tdc(tmp_path):
    path_to_data = RAW_DATA_DIR / "raw_test_data_01.tpx3"

    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    energy_estimation_test_parameters = np.load(path_to_energy_parameters)
    convert_tpx3_file(
        path_to_data,
        output_dir=tmp_path,
        estimate_energy=True,
        energy_estimation_parameters=energy_estimation_test_parameters,
        correct_timewalk=True,
        timewalk_b=167.0,
        timewalk_c=-0.016,
        tdc=True,
        verbose=True,
    )

    cdf = pd.read_parquet(tmp_path / "raw_test_data_01_cent.parquet")
    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
    assert required.issubset(cdf.columns)
    pd.testing.assert_frame_equal(cdf, pd.read_parquet(PROC_DATA_DIR / "raw_test_data_01_cent.parquet"), atol=0.01)

    # test that an empty tdc dataframe is still saved
    tdc_df = pd.read_parquet(tmp_path / "raw_test_data_01_tdc.parquet")
    required_tdc = {"tdc_t_ns", "tdc_type", "tdc_chip"}
    assert required_tdc.issubset(tdc_df.columns)
    assert len(tdc_df) == 0


def test_convert_tpx3_file_config(tmp_path):
    path_to_data = RAW_DATA_DIR / "raw_test_data_01.tpx3"
    path_to_yaml_config = CONFIG_DIR / "tpx3_configurations.yaml"

    with path_to_yaml_config.open() as f:
        data = yaml.safe_load(f)
    tpx3_config = Tpx3Config.model_validate(data)

    convert_tpx3_file(
        path_to_data,
        config=tpx3_config,
        output_dir=tmp_path,
    )
    processed_cdf_fpath = tmp_path / "raw_test_data_01_cent.parquet"
    cdf = pd.read_parquet(processed_cdf_fpath)
    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
    assert required.issubset(cdf.columns)
    tpx3_config.energy_estimation_parameters = None
    assert read_parquet_config(processed_cdf_fpath) == dict(tpx3_config)
    pd.testing.assert_frame_equal(cdf, pd.read_parquet(PROC_DATA_DIR / "raw_test_data_01_cent.parquet"), atol=0.01)


def test_convert_tpx3_files(tmp_path):
    path_to_data = RAW_DATA_DIR
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    raw_tpx3_file_paths = sorted([p for p in Path(path_to_data).glob("*") if p.is_file() and ".tpx3" in str(p)])
    convert_tpx3_files(
        raw_tpx3_file_paths,
        output_dir=tmp_path,
        estimate_energy=True,
        energy_estimation_parameters=path_to_energy_parameters,
        correct_timewalk=True,
        timewalk_b=167.0,
        timewalk_c=-0.016,
        verbose=True,
    )

    for i in range(len(raw_tpx3_file_paths)):
        cdf = pd.read_parquet(tmp_path / f"raw_test_data_{i:02d}_cent.parquet")
        required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
        assert required.issubset(cdf.columns)

    proc_parquet_files = sorted([p for p in Path(tmp_path).rglob("*") if p.is_file() and "cent.parquet" in str(p)])
    cur_proc_df = pd.concat([pd.read_parquet(f) for f in proc_parquet_files], ignore_index=True)

    # TODO remove this part when we are more confident in the refactor.
    # This processing data is from the old tpx3awkward, and
    # is being used as a "stable" output to compare against.
    path_to_proc_data = PROC_DATA_DIR / "processed_concat_test_data.parquet"
    stable_proc_df = pd.read_parquet(path_to_proc_data)

    pd.testing.assert_frame_equal(cur_proc_df, stable_proc_df, atol=0.01)


def test_convert_tpx3_files_parallel(tmp_path):
    path_to_data = RAW_DATA_DIR
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    raw_tpx3_file_paths = sorted([p for p in Path(path_to_data).glob("*") if p.is_file() and ".tpx3" in str(p)])
    convert_tpx3_files_parallel(
        raw_tpx3_file_paths,
        output_dir=tmp_path,
        estimate_energy=True,
        energy_estimation_parameters=path_to_energy_parameters,
        correct_timewalk=True,
        timewalk_b=167.0,
        timewalk_c=-0.016,
        verbose=True,
    )

    for i in range(len(raw_tpx3_file_paths)):
        cdf = pd.read_parquet(tmp_path / f"raw_test_data_{i:02d}_cent.parquet")
        required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
        assert required.issubset(cdf.columns)

    proc_parquet_files = sorted([p for p in Path(tmp_path).rglob("*") if p.is_file() and "cent.parquet" in str(p)])
    cur_proc_df = pd.concat([pd.read_parquet(f) for f in proc_parquet_files], ignore_index=True)

    # TODO remove this part when we are more confident in the refactor.
    # This processing data is from the old tpx3awkward, and
    # is being used as a "stable" output to compare against.
    path_to_proc_data = PROC_DATA_DIR / "processed_concat_test_data.parquet"
    stable_proc_df = pd.read_parquet(path_to_proc_data)

    pd.testing.assert_frame_equal(cur_proc_df, stable_proc_df, atol=0.01)
