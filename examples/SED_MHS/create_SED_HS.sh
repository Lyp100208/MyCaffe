#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../build/examples/SED_MHS
DATA=../../data/SED_MHS

echo "Creating leveldb..."

rm -rf sed_hs-train-leveldb
rm -rf sed_hs-test-leveldb

$EXAMPLES/convert_sed_hs_data.bin $DATA/pos.list 1 $DATA/neg.list 0 sed_hs-train-leveldb sed_hs-test-leveldb 0.8 3

echo "Done."
