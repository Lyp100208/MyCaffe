TOOLS=../../build/tools
DATA=../../data/Embrace

echo "Creating leveldb..."

rm -rf embrace_train_mean
rm -rf embrace_test_mean

GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin embrace-train-leveldb embrace_train_mean
GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin embrace-test-leveldb embrace_test_mean

echo "Done."