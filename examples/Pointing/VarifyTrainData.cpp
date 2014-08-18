#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <strstream>
#include <stdint.h>
#include <time.h>  

#include <sys/types.h> 
#include <sys/stat.h> 

#include <boost/shared_ptr.hpp>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

using namespace std;

#include "caffe/caffe.hpp"
#include "caffe/util/insert_splits.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
#define ROWS 32
#define COLS 32

int main( int argc, char** argv )
{
  // Initialize the leveldb
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;

  leveldb::Status status = leveldb::DB::Open(
      options, argv[1], &db_temp);


  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();

  // Read a data point, and use it to initialize the top blob.
  for (int i=0; i<200; i++) {
	  Datum datum;
	  datum.ParseFromString(iter_->value().ToString());

	  int ch = datum.channels();
	  int he = datum.height();
	  int wi = datum.width();
	  int count = datum.channels() * datum.height() * datum.width();

	  const string& data = datum.data();
	  IplImage* img = cvCreateImage( cvSize(wi, he), 8, 3 );

	  for (int h=0; h<he; h++){
	    unsigned char* p = (unsigned char*)( img->imageData + img->widthStep*h );
	    for (int w=0; w<wi; w++){
	      for (int c=0; c<ch; c++) {
	      	p[ ch*w + c ] = (unsigned char)data[ c*he*wi + h*wi + w ];
	      }
		}
	  }
	  cout<<datum.label()<<endl;
	  cvShowImage( "abc", img );
	  cvWaitKey( );
	  cvReleaseImage( &img );
	  iter_->Next();
  }

  return 0;
}