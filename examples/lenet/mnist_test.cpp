#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdint.h>
using namespace std;

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
  if (argc < 6) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto image_path label_path "
        << "[CPU/GPU]";
    return -1;
  }

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);

  if (argc == 6 && strcmp(argv[5], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[1], &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);


  // Open files
  std::ifstream image_file(argv[3], std::ios::in | std::ios::binary);
  std::ifstream label_file(argv[4], std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << argv[3];
  CHECK(label_file) << "Unable to open file " << argv[4];
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  label_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  label_file.read(reinterpret_cast<char*>(&num_labels), 4);
  num_labels = swap_endian(num_labels);
  CHECK_EQ(num_items, num_labels);
  image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);

  char label_0;
  unsigned char* pixels_0 = (unsigned char*)malloc(sizeof(unsigned char)*rows*cols);//(unsigned char*) (new char[rows * cols]);

  double test_accuracy = 0;
  vector<Blob<float>*> dummy_blob_input_vec;
  float* data = new float[ rows * cols ];
  float label[1];

  for (int itemid = 0; itemid < num_items; ++itemid) {
    image_file.read((char*)pixels_0, rows * cols);
    label_file.read(&label_0, 1);
    for (int i=0; i<rows*cols; i++){
      data[i] = (float)pixels_0[i] * 0.00390625;
      //cout<<data[i]<<" ";
      //printf("%d ", pixels_0[i]);
    }
    //cout<<"\n";
    label[0] = (float)label_0;
    //cout<<"label  = "<<label<<endl;
    //cin >> label_0;
    const vector<Blob<float>*>& result = caffe_test_net.MyForward( data,rows*cols, label, 1 );

    test_accuracy += result[0]->cpu_data()[0];

    //LOG(ERROR) << "accuracy: " << result[0]->cpu_data()[0];
  }
  LOG(ERROR) << "accuracy :" << test_accuracy/num_items;
  //delete []pixels_0;
  free( pixels_0 );
  delete []data;
  return 0;
}
