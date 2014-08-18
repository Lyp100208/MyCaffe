#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../build/examples/Embrace
DATA=../../data/Embrace

echo "Creating leveldb..."

rm -rf embrace-train-leveldb
rm -rf embrace-test-leveldb

$EXAMPLES/convert_sed_hs_data.bin $DATA/embrace_test_pos.list 1 $DATA/embrace_test_neg.list 0 embrace-test-leveldb 3
$EXAMPLES/convert_sed_hs_data.bin $DATA/embrace_train_pos.list 1 $DATA/embrace_train_neg.list 0 embrace-train-leveldb 3

echo "Done."
