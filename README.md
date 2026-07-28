# tpx3awkward

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![Coverage][coverage-badge]][coverage-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/NSLS2/tpx3awkward/workflows/CI/badge.svg
[actions-link]:             https://github.com/NSLS2/tpx3awkward/actions
[pypi-link]:                https://pypi.org/project/tpx3awkward/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/tpx3awkward
[pypi-version]:             https://img.shields.io/pypi/v/tpx3awkward
[coverage-badge]:           https://codecov.io/github/NSLS2/tpx3awkward/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/NSLS2/tpx3awkward

<!-- prettier-ignore-end -->

tpx3awkward is a Python package for efficient handling of data produced by the
Timepix family of detectors, this includes:

- Fast Decoding of raw `.tpx3` binary files
- Event clustering and centroiding
- Timewalk correction
- Energy estimation

## Installation

```bash
pip install tpx3awkward
```
