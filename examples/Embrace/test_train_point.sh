#!/usr/bin/env sh

TOOLS=../../build/examples/Embrace
NET=model_98_99/SED_point_test.prototxt
STATE=model_98_99/SED_embrace_quick_iter_14000
DETECT_FOLDER=../../examples/SED_HS/DetectionResult/train/a
VIDEO_FOLDER=/media/chenqi/BCB265B6B26575B4/TRECVID_VIDEO/VideoDev08
#GLOG_logtostderr=1 $TOOLS/test_cnn_result.bin model_98_99/SED_point_test.prototxt model_98_99/SED_point_quick_iter_20000 /home/chenqi/workspace/caffe-master/examples/SED_HS/DetectionResult/test /media/chenqi/BCB265B6B26575B4/TRECVID_VIDEO pointing_result_98_99
GLOG_logtostderr=1 $TOOLS/test_cnn_result_train.bin $NET $STATE $DETECT_FOLDER $VIDEO_FOLDER embrace_result_98_99