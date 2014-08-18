#!/usr/bin/env sh
EXAMPLES=../../build/examples/Pointing
DATA=../../data/Pointing

$EXAMPLES/get_feature_pic.bin $DATA/pointing_pos_test.list 1 $DATA/pointing_neg_test.list -1 feature/pointing_test.feature 3 feature/SED_point_test.prototxt feature/SED_point_quick_iter_69000 
$EXAMPLES/get_feature_pic.bin $DATA/pointing_pos_train.list 1 $DATA/pointing_neg_train.list -1 feature/pointing_train.feature 3 feature/SED_point_test.prototxt feature/SED_point_quick_iter_69000 