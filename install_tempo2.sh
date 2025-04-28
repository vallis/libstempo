#!/bin/bash -e

# default Tempo2 version
tempo2version="2021.07.1-correct"

usage() { echo "Usage: $0 [-p <install-path>] [-v <tempo2-version>]" 1>&2; exit 1; }

# default install location
prefix=$HOME/.local/
if [[ $# -eq 1 && "$1" != "-h" ]]; then
    # interpret single argument as install location
    prefix=$1
    echo 'Will install in' $prefix
else
	# allow arguments
	while getopts ":p:v:" o; do
    case "${o}" in
        p)
            prefix=${OPTARG}
            ;;
        v)
            tempo2version=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
fi

echo 'Will install in' $prefix
echo 'Will attempt to install tempo2 version ' $tempo2version

# make a destination directory for runtime files
export TEMPO2=$prefix/share/tempo2
mkdir -p $TEMPO2

curl -O https://bitbucket.org/psrsoft/tempo2/get/${tempo2version}.tar.gz
if [ $? -eq 0 ]; then
    echo 'Version '${tempo2version}' of Tempo2 does not exist. Please see, e.g., https://github.com/mattpitkin/tempo2/tags for a list of allowed version tags.'
    exit 1
fi
tar -zxvf ${tempo2version}.tar.gz

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
rm -rf ${tempo2version}.tar.gz
echo "Set TEMPO2 environment variable to ${TEMPO2} to make things run more smoothly."
