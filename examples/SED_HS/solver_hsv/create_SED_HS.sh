#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../../build/examples/SED_HS/solver_hsv
DATA=../../../data/SED_HS

echo "Creating leveldb..."

# rm -rf sed_hs-train-leveldb_gray
# rm -rf sed_hs-test-leveldb_gray

# $EXAMPLES/convert_sed_hs_data.bin $DATA/sed_hs_pos_train.list 1 $DATA/sed_hs_neg_train.list 0 sed_hs-train-leveldb_gray 1
# $EXAMPLES/convert_sed_hs_data.bin $DATA/sed_hs_pos_test.list 1 $DATA/sed_hs_neg_test.list 0 sed_hs-test-leveldb_gray 1

rm -rf sed_hs-train-leveldb
rm -rf sed_hs-test-leveldb

$EXAMPLES/convert_sed_hs_hsv_data.bin $DATA/sed_hs_pos_test.list 1 $DATA/sed_hs_neg_test.list 0 sed_hs-test-leveldb 3
$EXAMPLES/convert_sed_hs_hsv_data.bin $DATA/sed_hs_pos_train.list 1 $DATA/sed_hs_neg_train.list 0 sed_hs-train-leveldb 3

echo "Done."
