# libstempo

`libstempo` is a Python wrapper around the [tempo2](https://bitbucket.org/psrsoft/tempo2/src/master/) pulsar timing package.


## Installation

To use `libstempo2`, tempo2 must be installed as a prerequisite. Currently there are two recommended methods to do this.

1. Install via script. 
    ```bash
    curl -sSL https://raw.githubusercontent.com/vallis/libstempo/master/install_tempo2.sh | sh
    ```
    This will install the tempo2 library in a local directory (`$HOME/.local`). This method is recommended if you do not need to use tempo2 directly but just need the installation for `libstempo`.
2. Install via the [instructions](https://bitbucket.org/psrsoft/tempo2/src/master/README.md) on the tempo2 homepage. If this method is used, the `TEMPO2` environment variable will need to be set to use `libstempo`.

The `libstempo` package can be installed via `pip`:
```bash
pip install libstempo
```

## Usage

See [Demo Notebook 1](https://github.com/vallis/libstempo/blob/master/demo/libstempo-demo.ipynb) for basic usage and [Demo Notebook 2](https://github.com/vallis/libstempo/blob/master/demo/libstempo-toasim-demo.ipynb) for simulation usage.
