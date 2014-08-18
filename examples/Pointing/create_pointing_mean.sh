TOOLS=../../build/tools
DATA=../../data/Pointing

echo "Creating leveldb..."

rm -rf pointing_train_mean
rm -rf pointing_test_mean

GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin pointing-train-leveldb pointing_train_mean
GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin pointing-test-leveldb pointing_test_mean

echo "Done."