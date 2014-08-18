// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <iostream>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::cout;

namespace caffe {

template <typename Dtype>
void DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  //LOG(INFO)<<"(*top)[0]->num() = "<<(*top)[0]->num();
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
      cudaMemcpyHostToDevice));
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}
template <typename Dtype>
void DataLayer<Dtype>::MyForward_gpu(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top, 
      const Dtype* data, const int len_data, 
      const Dtype* label, const int len_label) {
  //LOG(INFO)<<"MyForward_gpu";
  CUDA_CHECK(
    cudaMemcpy(
      (*top)[0]->mutable_gpu_data(), data, sizeof(Dtype) * len_data, cudaMemcpyHostToDevice
      )
    );
  CUDA_CHECK(
    cudaMemcpy(
      (*top)[1]->mutable_gpu_data(), label, sizeof(Dtype) * len_label, cudaMemcpyHostToDevice
      )
    );
}
// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype DataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
