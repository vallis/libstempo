#!/bin/bash -e

# get install location
if [ $# -eq 0 ]
	then
		echo 'No install location defined, using' $HOME'/.local/'
		prefix=$HOME/.local/
	else
		prefix=$1
		echo 'Will install in' $prefix
fi

# make a destination directory for runtime files
export TEMPO2=$prefix/share/T2runtime
mkdir -p $TEMPO2

curl -O https://bitbucket.org/psrsoft/tempo2/get/2020.11.1.tar.gz
tar zxvf 2020.11.1.tar.gz

cd psrsoft-tempo2-*

./bootstrap
./configure --prefix=$prefix
make && make install
cp -r T2runtime $prefix/share
cd ..

rm -rf psrsoft-tempo2-*
rm -rf 2020.11.1.tar.gz
