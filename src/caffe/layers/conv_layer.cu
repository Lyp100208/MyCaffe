// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  // LOG(INFO)<<"NUM_ = "<<NUM_<<"\n"
  // <<" weight_offset = "<<weight_offset<<"\n"
  // <<" col_offset = "<<col_offset<<"\n"
  // <<" top_offset = "<<top_offset<<"\n"
  // <<" M_ = "<<M_<<"\n"
  // <<" K_ = "<<K_<<"\n"
  // <<" N_ = "<<N_
  // <<" CHANNELS_ = "<<CHANNELS_<<"\n"
  // <<" HEIGHT_ = "<<HEIGHT_<<"\n"
  // <<" WIDTH_ = "<<WIDTH_;
  for (int n = 0; n < NUM_; ++n) {
    // First, im2col
    im2col_gpu(bottom_data + bottom[0]->offset(n), CHANNELS_, HEIGHT_,
                      WIDTH_, KSIZE_, PAD_, STRIDE_, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < GROUP_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
    }
    // third, add bias
    if (biasterm_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0,
        sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n = 0; n < NUM_; ++n) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          1., bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  CUDA_CHECK(cudaMemset(weight_diff, 0,
      sizeof(Dtype) * this->blobs_[0]->count()));
  for (int n = 0; n < NUM_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_gpu(bottom_data + (*bottom)[0]->offset(n), CHANNELS_, HEIGHT_,
                      WIDTH_, KSIZE_, PAD_, STRIDE_, col_data);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < GROUP_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        weight_diff + weight_offset * g);
    }
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down) {
      for (int g = 0; g < GROUP_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff + col_offset * g);
      }
      // col2im back to the data
      col2im_gpu(col_diff, CHANNELS_, HEIGHT_, WIDTH_, KSIZE_, PAD_, STRIDE_,
          bottom_diff + (*bottom)[0]->offset(n));
    }
  }
  return Dtype(0.);
}


INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
