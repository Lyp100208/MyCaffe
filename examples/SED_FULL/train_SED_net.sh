#!/usr/bin/env sh

TOOLS=../../build/tools

#GLOG_logtostderr=1 $TOOLS/train_net.bin SED_HS_solver.prototxt
$TOOLS/train_net.bin SED_solver.prototxt