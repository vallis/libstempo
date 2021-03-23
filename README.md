# libstempo

`libstempo` is a Python wrapper around the [tempo2](https://bitbucket.org/psrsoft/tempo2/src/master/) pulsar timing package.


## Installation

To use `libstempo`, tempo2 must be installed as a prerequisite. Currently there are two recommended methods to do this.

1. Install via script. 
    ```bash
    curl -sSL https://raw.githubusercontent.com/vallis/libstempo/master/install_tempo2.sh | sh
    ```
    This will install the tempo2 library in a local directory (`$HOME/.local`). This method is recommended if you do not need to use tempo2 directly but just need the installation for `libstempo`.
    This script also takes in an option argument with a path to the
    install location. For example you could run:
    ```bash
    # need sudo if installing in a restricted location
    curl -sSL https://raw.githubusercontent.com/vallis/libstempo/master/install_tempo2.sh /usr/local | sudo sh -
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
