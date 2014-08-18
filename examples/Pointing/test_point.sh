#!/usr/bin/env sh

TOOLS=../../build/examples/Pointing

#GLOG_logtostderr=1 $TOOLS/test_cnn_result.bin model_98_99/SED_point_test.prototxt model_98_99/SED_point_quick_iter_20000 /home/chenqi/workspace/caffe-master/examples/SED_HS/DetectionResult/test /media/chenqi/BCB265B6B26575B4/TRECVID_VIDEO pointing_result_98_99
GLOG_logtostderr=1 $TOOLS/test_cnn_result.bin model_add_neg_98_99/SED_point_test.prototxt model_add_neg_98_99/SED_point_quick_iter_69000 /home/chenqi/workspace/caffe-master/examples/SED_HS/DetectionResult/test /media/chenqi/BCB265B6B26575B4/TRECVID_VIDEO pointing_result_add_neg_98_99