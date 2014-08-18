#!/usr/bin/env sh

TOOLS=../../build/examples/SED_HS

$TOOLS/get_feature.bin SED_HS_test_backup_feature.prototxt SED_HS_quick_iter_100000_best test_list_train_data.txt GPU 0.75 3
$TOOLS/get_feature.bin SED_HS_test_backup_feature.prototxt SED_HS_quick_iter_100000_best test_list.txt GPU 0.75 3