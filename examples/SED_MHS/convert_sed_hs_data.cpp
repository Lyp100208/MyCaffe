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
using namespace std;

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "caffe/proto/caffe.pb.h"

#define ROWS 32
#define COLS 32
#define USE_FRAME_NUM 2

class ImageUnit{
public:
	string path[6];
	int label;
};

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
bool load_image( ImageUnit& path_unit, const int rows, const int cols, const int channels, char* pixels )
{
  IplImage* img = NULL;
  int i,j,c;
  unsigned char* p;
  int img_step;

  for (int path_i=0; path_i<USE_FRAME_NUM; path_i++){
	  if (channels == 1)
		  img = cvLoadImage( path_unit.path[path_i].c_str(), 0 );
	  else
      img = cvLoadImage( path_unit.path[path_i].c_str() );
	  
	  if (img == NULL){
      return false;
	  }
	  img = size_normalize( img );
	  if (img == NULL){
      return false;
    }

    img_step = path_i*ROWS*COLS*channels;

    for ( i=0; i<ROWS; i++){
  		p = (unsigned char*)( img->imageData + img->widthStep*i );
  		for ( j=0; j<COLS; j++){
  		  for ( c=0; c<channels; c++){
  		    *( pixels + img_step + ROWS*COLS*c + i*COLS + j ) = p[j*channels + c];
  		  }
  		}
    }
	  cvReleaseImage( &img );
  }
  return true;
}

void mirror( const int rows, const int cols, const int channels, char* pixels )
{
  unsigned char* pixel_row = new unsigned char[ cols ];
  char* img;
  int i,j,c;
  for (c=0; c<channels; c++){
    img = pixels + rows*cols*c;
    for (i=0; i<rows; i++){
      for (j=0; j<cols; j++){
        pixel_row[j] = img[i*cols + j];
      }
      for (j=0; j<cols; j++){
        img[i*cols+j] = pixel_row[cols-j-1];
      }
    }
  }
  delete []pixel_row;
}

void convert_dataset(
  const char* pos_list, const int pos_label,
  const char* neg_list, const int neg_label,
  const char* db_train_filename, 
  const char* db_test_filename, 
  const float train_ratio, const int channels) {

  std::string str;
  int i=0,j;
  string folder_path;
  string line;
  int folder_size;
  vector< ImageUnit > train_path, test_path;

  // Open files pos
  std::ifstream image_file(pos_list, std::ios::in | std::ios::binary);
  if (!image_file){
  	cout<< "Unable to open file " << pos_list<<endl;
  	return;
  }
  std::vector< ImageUnit > pos_path;
  // Input file path
  int pos_num = 0;
  while( !image_file.eof() ){
  	folder_path = "";
    image_file>>folder_path;
    if (folder_path == "")
      break;
    image_file>>folder_size;

  	folder_path += "/";
  	folder_size /= 6;
    for (i=0; i<folder_size; i++){
      ImageUnit one;
      one.label = pos_label;
  		for (j=0; j<6; j++){
  			image_file>>line;
  			one.path[j] = folder_path + line;
  		}
      	pos_path.push_back( one ); 
      	pos_num++;
  	}
  }
  image_file.close();
  cout<<pos_path.size()<<"\n------------------------------------\n";
  //getchar();

  random_shuffle( pos_path.begin(), pos_path.end() );//将file_path中的数据随机打乱 
  train_path.assign( pos_path.begin(), pos_path.begin()+pos_path.size()*train_ratio );
  test_path.assign( pos_path.begin()+pos_path.size()*train_ratio+1, pos_path.end() );

  // Open files neg
  std::ifstream image_file2(neg_list, std::ios::in | std::ios::binary);
  if(!image_file2) {
  	cout<< "Unable to open file " << neg_list<<endl;
  	return;
  }
  std::vector< ImageUnit > neg_path;

  neg_path.clear();
  // Input file path
  while( !image_file2.eof() ){
  	folder_path = "";
    image_file2>>folder_path;
    if (folder_path == "")
      break;
    image_file2>>folder_size;

    folder_path += "/";
    folder_size /= 6;
    for (i=0; i<folder_size; i++){
    	ImageUnit one;
      one.label= neg_label;
    	for (j=0; j<6; j++){
    		image_file2>>line;
    		one.path[j] = folder_path + line;
    	}
      	neg_path.push_back( one ); 
    }
  }
  image_file2.close();

  cout<<neg_path.size()<<"\n------------------------------------\n";
  //getchar();

  random_shuffle( neg_path.begin(), neg_path.end() );
  int size,size2;
  if (neg_path.size() < train_path.size()*4){
     size = neg_path.size();
  }else{
     size = train_path.size()*4;
  }
  train_path.insert( train_path.end(), neg_path.begin(), neg_path.begin()+size );
  if (neg_path.size()-size < test_path.size()*4){
     size2 = neg_path.size()-size;
  }else{
     size2 = test_path.size()*4;
  }
  test_path.insert( test_path.end(), neg_path.begin()+size, neg_path.begin()+size+size2 );

  random_shuffle( train_path.begin(), train_path.end() );
  random_shuffle( test_path.begin(), test_path.end() );
 
  //Open leveldb
  leveldb::DB* train_db, *test_db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_train_filename, &train_db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_train_filename
      << ". Is it already existing?";


  char* pixels = new char[ROWS * COLS * channels * USE_FRAME_NUM ];
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;
  int num_items = train_path.size();

  caffe::Datum datum;
  datum.set_channels(channels*USE_FRAME_NUM);
  datum.set_height(ROWS);
  datum.set_width(COLS);

  std::cout << "A total of " << train_path.size()+train_path.size()-size << " items. pos = "<<(train_path.size()-size)*2<<" neg = "<<size<<std::endl;
  std::cout << "Rows: " << ROWS << " Cols: " << COLS<<" Channels: "<<channels<<"*"<<USE_FRAME_NUM<<std::endl;
  int itemid = 0;
  int key_id = 0;
  for ( itemid = 0; itemid < num_items; itemid++) {
    if ( load_image( train_path[itemid], ROWS, COLS, channels, pixels ) == true ){
      datum.set_data( pixels, ROWS*COLS*channels*USE_FRAME_NUM );
      datum.set_label( train_path[itemid].label );
      datum.SerializeToString( &value );
      snprintf(key, kMaxKeyLength, "%08d", key_id++);
      train_db->Put(leveldb::WriteOptions(), std::string(key), value);
      if (train_path[itemid].label == 0)
        continue;
      //mirror
      mirror( ROWS, COLS, channels*USE_FRAME_NUM, pixels );
      datum.set_data( pixels, ROWS*COLS*channels*USE_FRAME_NUM );
      datum.set_label( train_path[itemid].label );
      datum.SerializeToString( &value );
      snprintf(key, kMaxKeyLength, "%08d", key_id++);
      train_db->Put(leveldb::WriteOptions(), std::string(key), value);
    }else{
      LOG(INFO) << "Can't load image";
    }
  }  
  cout<<"------------------------------------\n";
  delete train_db;

  //getchar();
  //--------------------------------------------------------------------------------------------------------------------------
  status = leveldb::DB::Open(
      options, db_test_filename, &test_db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_test_filename
      << ". Is it already existing?";
  key_id = 0;
  num_items = test_path.size();
  std::cout << "A total of " << test_path.size()+test_path.size()-size2 << " items. pos = "<<(test_path.size()-size2)*2<<" neg = "<<size2<<std::endl;
  std::cout << "Rows: " << ROWS << " Cols: " << COLS<<" Channels: "<<channels*USE_FRAME_NUM<<std::endl;
  for ( itemid = 0; itemid < num_items; itemid++) {
    if ( load_image( test_path[itemid], ROWS, COLS, channels, pixels ) == true ){
      datum.set_data( pixels, ROWS*COLS*channels*USE_FRAME_NUM );
      datum.set_label( test_path[itemid].label );
      datum.SerializeToString( &value );
      snprintf(key, kMaxKeyLength, "%08d", key_id++);
      test_db->Put(leveldb::WriteOptions(), std::string(key), value);
      if (test_path[itemid].label == 0)
        continue;
      //mirror
      mirror( ROWS, COLS, channels*USE_FRAME_NUM, pixels );
      datum.set_data( pixels, ROWS*COLS*channels*USE_FRAME_NUM );
      datum.set_label( test_path[itemid].label );
      datum.SerializeToString( &value );
      snprintf(key, kMaxKeyLength, "%08d", key_id++);
      test_db->Put(leveldb::WriteOptions(), std::string(key), value);
    }else{
      LOG(INFO) << "Can't load image " ;
    }
  }

  cout<<"------------------------------------\n";
  //getchar();
  delete test_db;
  delete []pixels;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    printf("This script converts the SED_HS dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_mnist_data pos_list pos_label neg_list neg_label output_train_db_file output_test_db_file train_ratio channels\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], atoi( argv[2] ), argv[3], atoi( argv[4] ), argv[5],argv[6], atof(argv[7]), atoi(argv[8]) );
  }
  return 0;
}
