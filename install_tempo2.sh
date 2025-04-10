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
export TEMPO2=$prefix/share/tempo2
mkdir -p $TEMPO2

curl -O https://bitbucket.org/psrsoft/tempo2/get/2021.07.1-correct.tar.gz
tar zxvf 2021.07.1-correct.tar.gz

cd psrsoft-tempo2-*

# remove LT_LIB_DLLOAD from configure.ac
sed_in_place="-i ''" # For macOS
if [[ "$(uname -s)" == "Linux" ]]; then
  sed_in_place="-i"   # For Linux
fi
sed "$sed_in_place" "s/LT_LIB_DLLOAD//g" "configure.ac"

./bootstrap
./configure --prefix=$prefix
make && make install
cp -r T2runtime/* $TEMPO2
cd ..

rm -rf psrsoft-tempo2-*
rm -rf 2021.07.1-correct.tar.gz
echo "Set TEMPO2 environment variable to ${TEMPO2} to make things run more smoothly."
