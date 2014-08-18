#!/usr/bin/env sh

TOOLS=../../build/examples/SED_HS

$TOOLS/test_video_train_data.bin SED_HS_test_backup.prototxt SED_HS_quick_iter_100000_best test_list_train_data.txt GPU 0.75 3