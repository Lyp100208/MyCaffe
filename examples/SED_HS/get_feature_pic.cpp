#include <cuda_runtime.h>

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
#include "caffe/caffe.hpp"
#include "caffe/util/insert_splits.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
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
bool load_image( std::string& path, const int rows, const int cols, const int channels, float* pixels )
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

  float ave = 0;
  float tmp = 0;
  for ( i=0; i<ROWS; i++){
    p = (unsigned char*)( img->imageData + img->widthStep*i );
    for ( j=0; j<COLS; j++){
      for ( c=0; c<channels; c++){
      	tmp = (float)( p[j*channels+c]*0.00390625 );
        *( pixels + ROWS*COLS*c + i*COLS + j ) = tmp;
        ave += tmp;
      }
    }
  }
  int count = rows*cols*channels;
  ave /= count;
  for ( i=0; i<count; i++) {
  	pixels[i] -= ave;
  }
  cvReleaseImage( &img );
  return true;
}

void mirror( const int rows, const int cols, const int channels, float* pixels )
{
  float* pixel_row = new float[ cols*channels ];
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
  const char* filename, const int channels,
  const char* proto_file, const char* trained_file) {
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
  // random_shuffle( file_path.begin(), file_path.end() );
  // std::cout<<file_path.size()<<std::endl;
  // file_path.erase( file_path.begin(), file_path.begin()+file_path.size()*0.6 );
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
  // Open save file
  FILE* save_file = fopen( filename, "wb" );

  const int COUNT = ROWS*COLS*channels;
  float* data  = new float[ROWS * COLS * channels];
  float label = 1.0;
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;
  int num_items = file_path.size();

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);
  NetParameter test_net_param;
  ReadProtoFromTextFile(proto_file, &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(trained_file, &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  std::cout << "A total of " << num_items+pos_num << " items. pos = "<<pos_num*2<<" neg = "<<num_items-pos_num<<std::endl;
  std::cout << "Rows: " << ROWS << " Cols: " << COLS<<" Channels: "<<channels<<std::endl;

  int total = num_items+pos_num;
  fwrite( &total, sizeof(int), 1, save_file );
  for (int itemid = 0; itemid < num_items; itemid++) {
    str = file_path[itemid].first;
    if ( load_image( str, ROWS, COLS, channels, data ) == true ) {
      const vector<Blob<float>*>& feature = 
          caffe_test_net.MyForward_GetFeature( data, COUNT, &label, 1, "ip5");
      fwrite( &(file_path[itemid].second), sizeof(int), 1, save_file );
      fwrite( feature[0]->cpu_data(), sizeof(float), 64, save_file );
      if (file_path[itemid].second == 0)
        continue;
      //mirror
      mirror( ROWS, COLS, channels, data );
      const vector<Blob<float>*>& feature2 = 
          caffe_test_net.MyForward_GetFeature( data, COUNT, &label, 1, "ip5");
      fwrite( &(file_path[itemid].second), sizeof(int), 1, save_file );
      fwrite( feature2[0]->cpu_data(), sizeof(float), 64, save_file );
    } else {
      LOG(INFO) << "Can't load image :" << str;
    }
  }

  fclose( save_file );
  delete []data;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    printf(
           "Usage:\n"
           "    get_feature_pic pos_list pos_label neg_list neg_label output_file channels test_my.prototxt ***_quick_iter_**  \n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], atoi( argv[2] ), argv[3], atoi( argv[4] ), argv[5], atoi(argv[6]), argv[7], argv[8] );
  }
  return 0;
}
