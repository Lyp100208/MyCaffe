#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../build/examples/Pointing
DATA=../../data/Pointing

echo "Creating leveldb..."

rm -rf pointing-train-leveldb
rm -rf pointing-test-leveldb

$EXAMPLES/convert_sed_hs_data.bin $DATA/pointing_pos_test.list 1 $DATA/pointing_neg_test.list 0 pointing-test-leveldb 3
$EXAMPLES/convert_sed_hs_data.bin $DATA/pointing_pos_train.list 1 $DATA/pointing_neg_train.list 0 pointing-train-leveldb 3

echo "Done."
