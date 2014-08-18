TOOLS=../../build/tools
DATA=../../data/SED_HS

echo "Creating leveldb..."

rm -rf sed_hs_train_mean
rm -rf sed_hs_test_mean

GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin sed_hs-train-leveldb sed_hs_train_mean
GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin sed_hs-test-leveldb sed_hs_test_mean
# rm -rf sed_hs_train_mean_gray
# rm -rf sed_hs_test_mean_gray

# GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin sed_hs-train-leveldb_gray sed_hs_train_mean_gray
# GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin sed_hs-test-leveldb_gray sed_hs_test_mean_gray
echo "Done."