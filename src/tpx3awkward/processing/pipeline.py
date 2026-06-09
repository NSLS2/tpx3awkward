import gc
import logging
import multiprocessing
from functools import partial
from pathlib import Path

from tqdm import tqdm
from numpy.typing import NDArray
import numpy as np
import pandas as pd

from .cluster import cluster_decoded_df
from .config import Tpx3Config
from .corrections import estimate_energies, timewalk_corr, trim_corr
from .decoding import decode_tpx3_binary
from .files import converted_path, raw_as_numpy, save_df
from .schemas import empty_cent_df

logger = logging.getLogger(__name__)

def convert_tpx3_binary(tpx3_binary: NDArray[np.uint64], *, config: Tpx3Config | None = None, **overrides) -> pd.DataFrame:
    """
    Convert raw binary timepix3 events into decoded and centroided Pandas dataframes.

    Parameters
    ----------
    tpx3_binary : NDArray[np.uint64]
        Raw binary events formatted as 8 byte messages.
    config : Tpx3Config | None = None
        Defines the configurations for processing. If None, then will use `Tpx3Config.from_defaults` with overrides
    **overrides
        Used when config is None to override default parameters.
    """
    if config is not None and overrides:
        raise ValueError("Pass either `config` or keyword overrides, not both.")
    if config is None:
        config = Tpx3Config.from_defaults(**overrides)

    if config.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    decoded_df = decode_tpx3_binary(tpx3_binary)
    num_events = decoded_df.shape[0]

    if num_events == 0:
        logger.info("No events found!")
        gc.collect()
        return True

    logger.info(f"Decoding complete. {num_events} events found.")

    if config.estimate_energy:
        decoded_df["e"] = estimate_energies(
            decoded_df["x"].to_numpy(),
            decoded_df["y"].to_numpy(),
            decoded_df["ToT"].to_numpy(),
            config.energy_estimation_parameters,
        )

    if config.correct_timewalk:
        decoded_df["t_corr"] = timewalk_corr(
            decoded_df["t"].to_numpy(), decoded_df["ToT"].to_numpy(), config.timewalk_b, config.timewalk_c
        )

    if config.correct_trim:
        decoded_df = trim_corr(decoded_df, config.trim_mask)

    clustered_df = cluster_decoded_df(
        decoded_df,
        config.time_window,
        config.radius,
    )
    # maybe we should put this somewhere else...
    clustered_df.loc[clustered_df["xc"] >= 255.5, "xc"] += 2
    clustered_df.loc[clustered_df["yc"] >= 255.5, "yc"] += 2

    logger.info(f"Clustering and centroiding complete. Returning dataframe with {clustered_df.shape[0]} clustered events...")

    return clustered_df


def convert_tpx3_file(
    tpx3_fpath: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: Tpx3Config | None = None,
    **overrides,
):
    """
    Convert a .tpx3 file into decoded and centroided Pandas dataframes, which are stored in parquet or hdf5 files.

    Parameters
    ----------
    tpx3_fpath : str | Path
        Path to a `.tpx3` file
    output_dir : str | Path | None = None
        Directory to save converted files to. Will save to the same directory as file if None
    config : Tpx3Config | None = None
        Defines the configurations for processing. If None, then will use `Tpx3Config.from_defaults` with overrides
    **overrides
        Used when config is None to override default parameters.

    Raises
    ------
    FileNotFoundError
        If tpx3_fpath can't be found
    ValueError
        If the file doesn't have `.tpx3` suffix
    """
    if config is not None and overrides:
        raise ValueError("Pass either `config` or keyword overrides, not both.")
    if config is None:
        config = Tpx3Config.from_defaults(**overrides)

    if config.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    tpx3_fpath = Path(tpx3_fpath)

    if not tpx3_fpath.exists():
        raise FileNotFoundError(f"{tpx3_fpath} does not exist")
    if tpx3_fpath.suffix != ".tpx3":
        raise ValueError(f"{tpx3_fpath} is not a .tpx3 file")

    out_fpath = converted_path(tpx3_fpath, extension=config.file_extension, cent=False)
    cent_out_fpath = converted_path(tpx3_fpath, extension=config.file_extension, cent=True)

    if output_dir:
        output_dir = Path(output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"{output_dir} exists but is not a directory.")
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # TODO should we be doing this? compare with what sophy and/or pymepix does
        out_fpath = output_dir / out_fpath.name
        cent_out_fpath = output_dir / cent_out_fpath.name

    if out_fpath.exists() and cent_out_fpath.exists() and not config.overwrite:
        logger.info(f"-> {tpx3_fpath.name} already processed, skipping.")
        return False

    logger.info(f"-> Processing {tpx3_fpath.name}, size: {tpx3_fpath.stat().st_size / (1024 * 1024):.1f} MB")
    decoded_df = decode_tpx3_binary(raw_as_numpy(tpx3_fpath))
    num_events = decoded_df.shape[0]

    if num_events == 0:
        logger.info("No events found! Saving empty dataframes.")
        save_df(
            empty_cent_df(estimate_energy=config.estimate_energy, correct_timewalk=config.correct_timewalk), cent_out_fpath
        )
        gc.collect()
        return True

    logger.info(f"Loading {tpx3_fpath.name} complete. {num_events} events found.")

    if config.estimate_energy:
        decoded_df["e"] = estimate_energies(
            decoded_df["x"].to_numpy(),
            decoded_df["y"].to_numpy(),
            decoded_df["ToT"].to_numpy(),
            config.energy_estimation_parameters,
        )

    if config.correct_timewalk:
        decoded_df["t_corr"] = timewalk_corr(
            decoded_df["t"].to_numpy(), decoded_df["ToT"].to_numpy(), config.timewalk_b, config.timewalk_c
        )

    if config.correct_trim:
        decoded_df = trim_corr(decoded_df, config.trim_mask)

    clustered_df = cluster_decoded_df(
        decoded_df,
        config.time_window,
        config.radius,
    )
    # maybe we should put this somewhere else...
    clustered_df.loc[clustered_df["xc"] >= 255.5, "xc"] += 2
    clustered_df.loc[clustered_df["yc"] >= 255.5, "yc"] += 2

    logger.info(f"Clustering and centroiding complete. Saving to {cent_out_fpath.name}...")

    save_df(clustered_df, cent_out_fpath, config=config)
    logger.info(f"Saving {cent_out_fpath.name} complete. Checking file existence...")

    if cent_out_fpath.exists():
        logger.info(f"Confirmed {cent_out_fpath.name} exists!")
        to_return = True
    else:
        logger.info(f"WARNING: {cent_out_fpath.name} doesn't exist but it should?!")
        to_return = False

    del decoded_df, clustered_df
    gc.collect()
    return to_return


def convert_tpx3_files(
    tpx3_fpaths: list[str] | list[Path],
    *,
    output_dir: str | Path | None = None,
    config: Tpx3Config | None = None,
    **overrides,
):
    """
    Convert a list of .tpx3 files in a single process using convert_tpx3_file(), catching any errors.

    Parameters
    ----------
    tpx3_fpaths : list[str] | list[Path]
        .tpx3 file paths
    output_dir : str | Path | None = None
        Directory to save converted files to. Will save to the same directory as file if None
    config : Tpx3Config | None = None
        Defines the configurations for processing. If None, then will use `Tpx3Config.from_defaults` with overrides
    **overrides
        Used when config is None to override default parameters.
    """
    # create config here so it is only done once
    if config is not None and overrides:
        raise ValueError("Pass either `config` or keyword overrides, not both.")
    if config is None:
        config = Tpx3Config.from_defaults(**overrides)

    # Process files sequentially with tqdm progress bar
    for tpx3_fpath in tqdm(tpx3_fpaths, desc="Processing files"):
        try:
            convert_tpx3_file(tpx3_fpath, output_dir=output_dir, config=config)
        except Exception:  # noqa: PERF203
            logger.exception(f"Failed to process {tpx3_fpath}")


def convert_tpx3_file_worker(fpath, **kwargs):
    """Worker function for convert_tpx3_files_parallel in order to catch potential errors"""
    try:
        convert_tpx3_file(fpath, **kwargs)
        return True
    except Exception:
        logger.exception(f"Failed to process {fpath}")
        return False


def convert_tpx3_files_parallel(
    tpx3_fpaths: list[str] | list[Path],
    *,
    output_dir: str | Path | None = None,
    config: Tpx3Config | None = None,
    num_workers: int | None = None,
    **overrides,
):
    """
    Convert a list of .tpx3 files in parallel using multiprocessing and convert_tpx3_file().

    Parameters
    ----------
    tpx3_fpaths : list[str] | list[Path]
        .tpx3 file paths
    output_dir : str | Path | None = None
        Directory to save converted files to. Will save to the same directory as file if None
    config : Tpx3Config | None = None
        Defines the configurations for processing. If None, then will use `Tpx3Config.from_defaults` with overrides
    num_workers : int | None = None
        Number of worker processes to use. Defaults to max(1, (CPU count - 4)) to leave room for other tasks.
    **overrides
        Used when config is None to override default parameters.
    """
    # create config here so it is only done once
    if config is not None and overrides:
        raise ValueError("Pass either `config` or keyword overrides, not both.")
    if config is None:
        config = Tpx3Config.from_defaults(**overrides)

    if num_workers is None:
        max_workers = min(multiprocessing.cpu_count() - 4, len(tpx3_fpaths))  # Leave 4 cores free
    else:
        max_workers = min(num_workers, len(tpx3_fpaths))  # Don't use more workers than files

    max_workers = max(max_workers, 1)

    worker_func = partial(convert_tpx3_file_worker, output_dir=output_dir, config=config)

    with multiprocessing.Pool(processes=max_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_func, tpx3_fpaths),
                total=len(tpx3_fpaths),
                desc="Processing files",
            )
        )

    # Count successes
    num_true = sum(results)
    print(f"Successfully converted {num_true} out of {len(tpx3_fpaths)}!")
