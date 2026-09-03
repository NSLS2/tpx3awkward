import numpy as np
import pandas as pd


def empty_raw_df() -> pd.DataFrame:
    """
    Create an empty event DataFrame with the expected columns from _ingest_raw_data()
    and the specified data types.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with columns:
        ['x', 'y', 'ToT', 't', 'chip'] and appropriate dtypes
    """
    data = {
        "x": np.array([], dtype="u2"),  # uint16
        "y": np.array([], dtype="u2"),  # uint16
        "ToT": np.array([], dtype="u4"),  # uint32
        "t": np.array([], dtype="u8"),  # uint64
        "chip": np.array([], dtype="u1"),  # uint8
    }

    return pd.DataFrame(data)


def empty_tdc_df() -> pd.DataFrame:
    """
    Create an empty tdc DataFrame with the expected columns from _ingest_raw_data()
    and the specified data types.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with columns:
        ['tdc_t_ns', 'tdc_type', 'tdc_chip'] and appropriate dtypes
    """
    data = {
        "tdc_t_ns": np.array([], dtype="f8"),
        "tdc_type": np.array([], dtype=np.uint8),
        "tdc_chip": np.array([], dtype=np.uint8),
    }

    return pd.DataFrame(data)


def empty_cent_df(estimate_energy: bool = False, correct_timewalk: bool = False) -> pd.DataFrame:
    """
    Create an empty DataFrame with the expected columns from ingest_cent_data()
    and the specified data types.

    Parameters
    ----------
    estimate_energy : bool, optional
        Whether to include the 'e_sum' column (energy estimates). Default is False.
    correct_timewalk : bool, optional
        Whether to include the 't_corr' column (timewalk correction). Default is False


    Returns
    -------
    pd.DataFrame
        Empty DataFrame with columns:
        ['t', 'xc', 'yc', 'ToT_max', 'ToT_sum', 'e_sum', 'n'] and appropriate dtypes
    """
    data = {
        "t": np.array([], dtype="uint64"),  # uint64
        "xc": np.array([], dtype="float32"),  # float32
        "yc": np.array([], dtype="float32"),  # float32
        "ToT_max": np.array([], dtype="uint32"),  # uint32
        "ToT_sum": np.array([], dtype="uint32"),  # uint32
        "n": np.array([], dtype="u1"),  # uint8 (ubyte)
    }

    if estimate_energy:
        data["e_sum"] = np.array([], dtype="float32")
    if correct_timewalk:
        data["t_corr"] = np.array([], dtype="uint64")

    return pd.DataFrame(data)
