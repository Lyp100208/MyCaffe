#!/usr/bin/env sh

TOOLS=../../build/examples/Embrace

NET=model_97_98/SED_point_test.prototxt
STATE=model_97_98/SED_embrace_quick_iter_51000
DETECT_FOLDER=/home/chenqi/workspace/caffe-master/examples/SED_HS/DetectionResult/test
VIDEO_FOLDER=/media/chenqi/BCB265B6B26575B4/TRECVID_VIDEO
#GLOG_logtostderr=1 $TOOLS/test_cnn_result.bin model_97_98/SED_point_test.prototxt model_97_98/SED_point_quick_iter_20000 /home/chenqi/workspace/caffe-master/examples/SED_HS/DetectionResult/test /media/chenqi/BCB265B6B26575B4/TRECVID_VIDEO pointing_result_97_98
GLOG_logtostderr=1 $TOOLS/test_cnn_result.bin $NET $STATE $DETECT_FOLDER $VIDEO_FOLDER embrace_result_97_98