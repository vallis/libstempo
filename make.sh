#!/bin/sh

git checkout master -- libstempo-demo.ipynb
git checkout master -- libstempo-toasim-demo.ipynb
ipython nbconvert libstempo-demo.ipynb --to html --template github
ipython nbconvert libstempo-toasim-demo.ipynb --to html --template github
