// Copyright Yangqing Jia 2013
//
// This script converts the MNIST dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_mnist_data input_image_file input_label_file output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <iterator>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "caffe/proto/caffe.pb.h"

#define ROWS 32
#define COLS 32

IplImage* size_normalize( IplImage* img )
{
  int width, height;
  CvRect rect;

  if( img == NULL )
    return NULL;
  width = img->width;
  height = img->height;

  if ( width < height ){
    rect.x = 0;
    rect.width = width;
    rect.y = ( height - width )/2;
    rect.height = width;

    cvSetImageROI( img, rect );
  }else if ( height < width ){
    rect.x = ( width - height )/2;
    rect.width = height;
    rect.y = 0;
    rect.height = height;

    cvSetImageROI( img, rect );
  }

  if ( width != COLS ){
    IplImage* tmp = cvCreateImage( cvSize( COLS, ROWS ), 8, img->nChannels);

    if (tmp == NULL)
      return NULL;

    cvResize( img, tmp );
    cvReleaseImage( &img );
    img = NULL;

    return tmp;
  }else{
    return img;
  }
}
bool load_image( std::string& path, const int rows, const int cols, const int channels, char* pixels )
{
  IplImage* img = NULL;
  int i,j,c;
  unsigned char* p;

  if (channels == 1)
    img = cvLoadImage( path.c_str(), 0 );
  else
    img = cvLoadImage( path.c_str() );
  
  if (img == NULL){
    return false;
  }
  img = size_normalize( img );
  if (img == NULL){
    return false;
  }

  for ( i=0; i<ROWS; i++){
    p = (unsigned char*)( img->imageData + img->widthStep*i );
    for ( j=0; j<COLS; j++){
      for ( c=0; c<channels; c++){
        *( pixels + ROWS*COLS*c + i*COLS + j ) = p[j*channels + c];
      }
    }
  }
  cvReleaseImage( &img );
  return true;
}

void mirror( const int rows, const int cols, const int channels, char* pixels )
{
  unsigned char* pixel_row = new unsigned char[ cols*channels ];
  int i,j,c;

  for (i=0; i<rows; i++){
    for (j=0; j<cols; j++){
      for (c=0; c<channels; c++){
        pixel_row[ cols*c + j] = pixels[rows*cols*c + cols*i + j];
      }
    }
    for (j=0; j<cols; j++){
      for (c=0; c<channels; c++)
        pixels[rows*cols*c + cols*i + j] = pixel_row[ cols*c + cols - j -1 ];
    }
  }
  delete []pixel_row;
}

void convert_dataset(
  const char* pos_list, const int pos_label,
  const char* neg_list, const int neg_label,
  const char* db_filename, const int channels) {
  // Open files neg
  std::ifstream image_file2(neg_list, std::ios::in | std::ios::binary);
  CHECK(image_file2) << "Unable to open file " << neg_list;
  std::vector< std::pair<std::string, int> > file_path;
  std::string str;
  
  file_path.clear();
  // Input file path
  while( !image_file2.eof() ){
    getline( image_file2, str );
    if (str == "")
      break;
    file_path.push_back( std::make_pair<std::string,int>(str,neg_label) ); 
  }
  image_file2.close();
  random_shuffle( file_path.begin(), file_path.end() );
  // std::cout<<file_path.size()<<std::endl;
  // file_path.erase( file_path.begin(), file_path.begin()+file_path.size()*0.3 );
  // std::cout<<file_path.size()<<std::endl;
  // Open files pos
  std::ifstream image_file(pos_list, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << pos_list;
  
  // Input file path
  int pos_num = 0;
  while( !image_file.eof() ){
    getline( image_file, str );
    if (str == "")
      break;
    file_path.push_back( std::make_pair<std::string,int>(str,pos_label) ); 
    pos_num++;
  }
  image_file.close();

  random_shuffle( file_path.begin(), file_path.end() );//将file_path中的数据随机打乱
  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

  char* pixels = new char[ROWS * COLS * channels];
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;
  int num_items = file_path.size();

  caffe::Datum datum;
  datum.set_channels(channels);
  datum.set_height(ROWS);
  datum.set_width(COLS);

  std::cout << "A total of " << num_items << " items. pos = "<<pos_num<<" neg = "<<num_items-pos_num<<std::endl;
  std::cout << "Rows: " << ROWS << " Cols: " << COLS<<" Channels: "<<channels<<std::endl;
  int itemid = 0;
  int key_id = 0;
  for (int itemid = 0; itemid < num_items; itemid++) {
    str = file_path[itemid].first;
    if ( load_image( str, ROWS, COLS, channels, pixels ) == true ){
      datum.set_data( pixels, ROWS*COLS*channels );
      datum.set_label( file_path[itemid].second );
      datum.SerializeToString( &value );
      snprintf(key, kMaxKeyLength, "%08d", key_id++);
      db->Put(leveldb::WriteOptions(), std::string(key), value);
      // if (file_path[itemid].second == 0)
      //   continue;
      // //mirror
      // mirror( ROWS, COLS, channels, pixels );
      // datum.set_data( pixels, ROWS*COLS*channels );
      // datum.set_label( file_path[itemid].second );
      // datum.SerializeToString( &value );
      // snprintf(key, kMaxKeyLength, "%08d", key_id++);
      // db->Put(leveldb::WriteOptions(), std::string(key), value);
    }else{
      LOG(INFO) << "Can't load image :" << str;
    }
  }

  delete db;
  delete pixels;
}

int main(int argc, char** argv) {
  if (argc != 7) {
    printf("This script converts the SED_HS dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_mnist_data pos_list pos_label neg_list neg_label output_db_file channels\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], atoi( argv[2] ), argv[3], atoi( argv[4] ), argv[5], atoi(argv[6]) );
  }
  return 0;
}
