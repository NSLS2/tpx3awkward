from pathlib import Path

import pandas as pd

from tpx3awkward.processing.decoding import decode_tpx3_binary
from tpx3awkward.processing.files import raw_as_numpy

RAW_DATA_DIR = Path(__file__).parents[1] / "data/raw"


def test_decode_tpx3_binary_missing_messages(capsys):
    path_to_raw_data = RAW_DATA_DIR / "raw_test_data_00.tpx3"
    binary = raw_as_numpy(path_to_raw_data)
    decode_tpx3_binary(binary)
    decode_tpx3_binary_capture = capsys.readouterr()
    assert "Missing messages!" not in decode_tpx3_binary_capture.out


def test_decode_tpx3_binary_serval_4_missing_messages(capsys):
    path_to_raw_data = RAW_DATA_DIR / "serval_4_3/raw_test_data_serval_4_3_0.tpx3"
    binary = raw_as_numpy(path_to_raw_data)
    decode_tpx3_binary(binary)
    decode_tpx3_binary_capture = capsys.readouterr()
    # numba doesn't support raising errors, so we print error messages
    assert "Missing messages!" not in decode_tpx3_binary_capture.out


def test_decode_tpx3_binary_tdc():
    path_to_raw_data = RAW_DATA_DIR / "tdc/"
    raw_tpx3_file_paths = sorted([p for p in path_to_raw_data.glob("*") if p.is_file() and ".tpx3" in str(p)])

    tdc_dfs = []
    for raw_tpx3_file_path in raw_tpx3_file_paths:
        binary = raw_as_numpy(raw_tpx3_file_path)
        _, tdc_df = decode_tpx3_binary(binary)
        tdc_dfs.append(tdc_df)

    concat_tdc_df = pd.concat(tdc_dfs, ignore_index=True)
    required = {"tdc_t_ns", "tdc_type", "tdc_chip"}
    assert required.issubset(concat_tdc_df.columns)

    # time should be monotonically increasing
    # tdc events come in bunches per chip, so only check for one
    concat_tdc_df_chip_0 = concat_tdc_df.loc[concat_tdc_df["tdc_chip"] == 0]
    assert concat_tdc_df_chip_0["tdc_t_ns"].is_monotonic_increasing
