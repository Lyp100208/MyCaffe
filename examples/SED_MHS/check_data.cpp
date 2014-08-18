// Copyright 2013 Yangqing Jia

#include <glog/logging.h>
#include <leveldb/db.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <iostream>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::max;
using std::cin;
using std::cout;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    LOG(ERROR) << "Usage: compute_image_mean input_leveldb output_file";
    return(0);
  }

  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = false;

  LOG(INFO) << "Opening leveldb " << argv[1];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[1], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[1];

  leveldb::ReadOptions read_options;
  read_options.fill_cache = false;
  leveldb::Iterator* it = db->NewIterator(read_options);
  it->SeekToFirst();
  Datum datum;
  BlobProto sum_blob;
  int count = 0;
  datum.ParseFromString(it->value().ToString());
  LOG(INFO)<<"datum.size = "<<datum.data().size();
  LOG(INFO)<<"datum.channels = "<<datum.channels();
  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  int width = datum.width(), height = datum.height();
  int channels = datum.channels();
  IplImage* img = cvCreateImage( cvSize(width, height), 8,  3 );
  IplImage* img2= cvCreateImage( cvSize(width*5, height*5), 8, 3 );

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    // just a dummy operation
    datum.ParseFromString(it->value().ToString());
    const string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(), datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      std::cout<<datum.label()<<std::endl;
      for (int k=0; k<channels; k++){
       // for (int i=0; i<height*width; i++){
       //    ( (uint8_t*)(img->imageData+img->widthStep*(i/width)) )[ i%width ] = (uint8_t)data[k*width*height+i];
       //  }
       //  cvResize( img , img2 );
       //  cvShowImage("img", img2);
       //  cvWaitKey(0);
        if (k<3){
          for (int i=0; i<height*width; i++){
            ( (uint8_t*)(img->imageData+img->widthStep*(i/width)) )[ i%width*3+k ] = (uint8_t)data[k*width*height+i];
          }
        }else{
          for (int i=0; i<height*width; i++){
            ( (uint8_t*)(img->imageData+img->widthStep*(i/width)) )[ i%width*3+k-3 ] = (uint8_t)data[k*width*height+i];
          }
        }
        if (k%3==2){
          cvResize( img , img2 );
          cvShowImage("img", img2);
          cvWaitKey(0);
        }
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  cvReleaseImage( &img2 );
  cvReleaseImage( &img );
  delete db;
  return 0;
}
