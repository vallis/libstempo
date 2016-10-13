## libstempo — a Python wrapper for tempo2 ##

Please see http://vallis.github.io/libstempo.

Note (2016/10/12): v2.3.0 (or higher) of libstempo requires tempo2 from fall 2016 (or newer). tempo2 changed the internal API considerably, which required corresponding changes in libstempo. Starting with this version, libstempo provides its own least-squares fit. If you don't have tempo2 installed, libstempo will attempt to download it, compile it, and install it.

Older note: the master branch has now been switched to version 2.X of libstempo, which has a new (and better!) API. Look at the [demo notebook](https://github.com/vallis/libstempo/blob/master/demo/libstempo-demo.ipynb) for a description. The old API is still available in the libstempo1 branch.
