#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../build/examples/SED_FULL
DATA=../../data/SED_FULL

echo "Creating leveldb..."

# rm -rf sed_full-train-leveldb_gray
# rm -rf sed_full-test-leveldb_gray

# $EXAMPLES/convert_sed_data.bin $DATA/sed_full_pos_train.list 1 $DATA/sed_full_neg_train.list 0 sed_full-train-leveldb_gray 1
# $EXAMPLES/convert_sed_data.bin $DATA/sed_full_pos_test.list 1 $DATA/sed_full_neg_test.list 0 sed_full-test-leveldb_gray 1

# rm -rf sed_full-train-leveldb
# rm -rf sed_full-test-leveldb

# $EXAMPLES/convert_sed_data.bin $DATA/sed_full_pos_test.list 1 $DATA/sed_full_neg_test.list 0 sed_full-test-leveldb 3
# $EXAMPLES/convert_sed_data.bin $DATA/sed_full_pos_train.list 1 $DATA/sed_full_neg_train.list 0 sed_full-train-leveldb 3

rm -rf sed_full-train-leveldb-crop
rm -rf sed_full-test-leveldb-crop

$EXAMPLES/convert_sed_data.bin $DATA/sed_full_pos_test.list 1 $DATA/sed_full_neg_test.list 0 sed_full-test-leveldb-crop 3
$EXAMPLES/convert_sed_data.bin $DATA/sed_full_pos_train.list 1 $DATA/sed_full_neg_train.list 0 sed_full-train-leveldb-crop 3

echo "Done."
