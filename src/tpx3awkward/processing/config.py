from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Tpx3Config(BaseModel):
    """
    Configuration for TPX3 data processing pipeline.

    This includes clustering, corrections, energy estimations, and output formats.

    Attributes
    ----------
    time_window : float
        Temporal clustering window in microseconds.

    radius : float
        Spatial clustering radius.

    estimate_energy : bool, default=False
        Enable energy estimation during processing.

    energy_estimation_parameters : numpy.ndarray | Path | str | None, default=None
        Energy estimation calibration parameters. May be provided directly as a
        NumPy array of shape (512, 512, 4) or as a path to a ``.npy`` file
        containing such an array.

    correct_timewalk : bool, default=False
        Enable timewalk correction.

    timewalk_b : float | None, default=None
        Timewalk correction parameter ``b`` from the exp. decay fit.

    timewalk_c : float | None, default=None
        Timewalk correction parameter ``c`` from the exp. decay fit.

    correct_trim : bool, default=False
        Enable trim correction.

    trim_mask : numpy.ndarray | Path | str | None, default=None
        Trim correction mask. May be provided directly as a NumPy
        array or as a path to a ``.npy`` file.

    tdc : bool, default = False
        Enable processing and saving of TDC packets into a separate dataframe.

    file_extension : str, default=".parquet"
        Output file extension. Must be either ``".h5"`` or ``".parquet"``.

    add_centroid_cols : bool, default=True
        Whether to include centroid columns in the output dataframe.

    overwrite : bool, default=True
        Whether existing output files should be overwritten if they already exist in
        the output directory.

    verbose : bool, default=False
        Enable verbose logging output.

    Raises
    ------
    ValueError
        Thrown for invalid file paths, unsupported values, and configuration compatability issues.
    TypeError
        Thrown when input values don't match the expected type.

    Examples
    --------
    Define a Tpx3Config object for a pipeline function call:

    >>> from tpx3awkward import Tpx3Config
    >>> tpx3config = Tpx3Config(time_window=0.3, radius=3, ..., verbose=False)

    Notes
    -----
    Paths provided for ``energy_estimation_parameters`` and
    ``trim_mask`` are automatically loaded using ``numpy.load()``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # --- Clustering ---
    time_window: float
    radius: float

    # --- Energy estimation ---
    estimate_energy: bool = False
    energy_estimation_parameters: np.ndarray | None = None

    # --- Timewalk ---
    correct_timewalk: bool = False
    timewalk_b: float | None = None
    timewalk_c: float | None = None

    # --- Trim correction ---
    correct_trim: bool = False
    trim_mask: np.ndarray | None = None

    # --- Misc. ---
    tdc: bool = False
    file_extension: str = ".parquet"
    add_centroid_cols: bool = True
    overwrite: bool = True
    verbose: bool = False

    @field_validator("energy_estimation_parameters", mode="before")
    @classmethod
    def load_energy_estimation_parameters(cls, value: np.ndarray | Path | str):
        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            try:
                return np.load(path)
            except Exception as e:
                raise ValueError(f"Failed to load energy estimation parameters with path {path}") from e
        raise TypeError("energy_estimation_parameters must be a numpy array, a path, or None")

    @field_validator("trim_mask", mode="before")
    @classmethod
    def load_trim_mask(cls, value: np.ndarray | Path | str):
        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            try:
                return np.load(path)
            except Exception as e:
                raise ValueError(f"Failed to load trim mask with path {path}") from e
        raise TypeError("trim_mask must be a numpy array, a path, or None")

    @field_validator("file_extension", mode="after")
    @classmethod
    def validate_file_extension(cls, value: str):
        if value not in (".h5", ".parquet"):
            raise ValueError("file_extension must be one of '.h5' or '.parquet'")
        return value

    @model_validator(mode="after")
    def validate_dependencies(self):
        if self.estimate_energy and self.energy_estimation_parameters is None:
            raise ValueError("energy_estimation_parameters must be provided when estimate_energy=True")
        if self.correct_timewalk and (self.timewalk_b is None or self.timewalk_c is None):
            raise ValueError("timewalk_b and timewalk_c parameters must be provided when correct_timewalk=True")
        if self.correct_trim and (self.trim_mask is None):
            raise ValueError("trim_mask must be provided when correct_trim=True")
        return self

    @classmethod
    def from_defaults(cls, **overrides) -> "Tpx3Config":
        defaults = {
            "time_window": 0.3,
            "radius": 3,
        }

        defaults.update(overrides)
        return cls(**defaults)
