# libstempo

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/vallis/libstempo)](https://github.com/vallis/libstempo/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/libstempo)](https://pypi.org/project/libstempo/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/libstempo.svg)](https://anaconda.org/conda-forge/libstempo)
[![libstempo CI tests](https://github.com/vallis/libstempo/actions/workflows/ci_tests.yml/badge.svg)](https://github.com/vallis/libstempo/actions/workflows/ci_tests.yml)


[![Python Versions](https://img.shields.io/badge/python-3.6%2C%203.7%2C%203.8%2C%203.9-blue.svg)]()
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/vallis/libstempo/blob/master/LICENSE)

`libstempo` is a Python wrapper around the [tempo2](https://bitbucket.org/psrsoft/tempo2/src/master/) pulsar timing package.


## Installation

### conda Install

`libstempo` is installed most simply via [conda](https://docs.conda.io/en/latest/) as the `tempo` dependency
is bundled in the conda recipe. Simply use
```bash
conda install -c conda-forge libstempo
```

### pip Install

To use `libstempo` with pip (or from source), tempo2 must be installed as a prerequisite. Currently there are two recommended methods to do this.

1. Install via script. 
    ```bash
    curl -sSL https://raw.githubusercontent.com/vallis/libstempo/master/install_tempo2.sh | sh
    ```
    This will install the tempo2 library in a local directory (`$HOME/.local`). This method is recommended if you do not need to use tempo2 directly but just need the installation for `libstempo`. You can also set the path to the install location. For example, to install in `/usr/local`, you could run:
    ```bash
    # need sudo if installing in a restricted location
    curl -sSL https://raw.githubusercontent.com/vallis/libstempo/master/install_tempo2.sh | sudo sh -s /usr/local
    ``` 
2. Install via the [instructions](https://bitbucket.org/psrsoft/tempo2/src/master/README.md) on the tempo2 homepage. If this method is used, the `TEMPO2` environment variable will need to be set to use `libstempo`.

In either case, it is best practice to set the `TEMPO2` environment
variable so that it can be easily discovered by `libstempo`.

The `libstempo` package can be installed via `pip`:
```bash
pip install libstempo
```

To use `astropy` for units:
```bash
pip install libstempo[astropy]
```

If you have installed `tempo2` in a location that is not in your path or not the default from `install_tempo2.sh`, you will need to install 
`libstempo` with an environment variable (e.g. if `tempo2` is in `/opt/local/bin`)
```bash
TEMPO2_PREFIX=/opt/local pip install libstempo
```
or
```bash
export TEMPO2_PREFIX=/opt/local
pip install libstempo
```

## Usage

See [Demo Notebook 1](https://github.com/vallis/libstempo/blob/master/demo/libstempo-demo.ipynb) for basic usage and [Demo Notebook 2](https://github.com/vallis/libstempo/blob/master/demo/libstempo-toasim-demo.ipynb) for simulation usage.
