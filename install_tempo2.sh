#!/bin/bash -e

# get install location
if [ $# -eq 0 ]
	then
		echo 'No install location defined, using' $HOME'/.local/tempo2'
		prefix=$HOME/.local/tempo2
	else
		prefix=$1
		echo 'Will install in' $prefix
fi

# make a destination directory for runtime files
export TEMPO2=$prefix/share/tempo2
mkdir -p $TEMPO2

# git clone https://bitbucket.org/psrsoft/tempo2.git
# git clone https://jellis11@bitbucket.org/jellis11/tempo2.git

curl -O https://bitbucket.org/psrsoft/tempo2/get/master.tar.gz -z master.tar.gz
# curl -O https://bitbucket.org/jellis11/tempo2/get/master.tar.gz -z master.tar.gz
tar zxvf master.tar.gz

cd psrsoft-tempo2-*
# cd jellis11-tempo2-*

./bootstrap
./configure --prefix=$prefix
make && make install
# make plugins-install
cp -Rp T2runtime/* $TEMPO2/.
cd ..

rm -rf psrsoft-tempo2-*
# rm -rf jellis11-tempo2-*
