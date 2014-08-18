#!/usr/bin/env sh
EXAMPLES=../../build/examples/SED_HS
DATA=../../data/SED_HS

$EXAMPLES/get_feature_pic.bin $DATA/sed_hs_pos_test.list 1 $DATA/sed_hs_neg_test.list 0 sed_hs_test.feature 3 SED_HS_test_backup_feature2.prototxt SED_HS_quick_iter_100000_best 
$EXAMPLES/get_feature_pic.bin $DATA/sed_hs_pos_train.list 1 $DATA/sed_hs_neg_train.list 0 sed_hs_train.feature 3 SED_HS_test_backup_feature2.prototxt SED_HS_quick_iter_100000_best 