#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../build/examples/lenet
DATA=../../data/mnist
#DATA=../../data/NICTA

echo "Creating leveldb..."

rm -rf mnist-train-leveldb
rm -rf mnist-test-leveldb
# rm -rf nicta-train-leveldb
# rm -rf nicta-test-leveldb

$EXAMPLES/convert_mnist_data.bin $DATA/train-images-idx3-ubyte $DATA/train-labels-idx1-ubyte mnist-train-leveldb
$EXAMPLES/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte $DATA/t10k-labels-idx1-ubyte mnist-test-leveldb

# $EXAMPLES/convert_mnist_data.bin $DATA/train_images_idx3_ubyte.data $DATA/train_labels_idx1_ubyte.data nicta-train-leveldb
# $EXAMPLES/convert_mnist_data.bin $DATA/test_images_idx3_ubyte.data $DATA/test_labels_idx1_ubyte.data nicta-test-leveldb

echo "Done."
