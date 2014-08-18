#!/usr/bin/env sh

TOOLS=../../build/examples/lenet
DATA=../../data/mnist

GLOG_logtostderr=1 $TOOLS/mnist_test.bin lenet_test_my.prototxt lenet_iter_8000  $DATA/t10k-images-idx3-ubyte $DATA/t10k-labels-idx1-ubyte  GPU
