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

class Proposal{
public:
  Proposal(){
    id = -1;
    num = 0;
  }
  bool operator > (const Proposal &m)const {
                return score > m.score;
  }
  bool operator < (const Proposal &m)const {
                return score < m.score;
  }
public:
  shared_ptr<float> data;
  int x;
  int y;
  int width;
  int height;
  int id;
  int num;
  float score;
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
    IplImage* tmp = cvCreateImage( cvSize( COLS, ROWS ), 8, 3);

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

void show(  std::string& path ){
  IplImage* img = cvLoadImage( path.c_str() );

  if (img == NULL){
    return ;
  }    

  img = size_normalize( img );
  if (img == NULL){
    return ;
  }
  cvShowImage( "img", img );
  cvWaitKey( 0 );  
  cvReleaseImage( &img );
}

bool load_image( std::string& path, const int rows, const int cols, unsigned char* pixels )
{
  IplImage* img = cvLoadImage( path.c_str() );

  if (img == NULL){
    return false;
  }    

  img = size_normalize( img );
  if (img == NULL){
    return false;
  }

  for (int i=0; i<ROWS; i++){
    unsigned char* p = (unsigned char*)( img->imageData + img->widthStep*i );
    for (int j=0; j<COLS; j++){
      *( pixels + ROWS*COLS*0 + i*COLS + j ) = p[j*3 + 0];
      *( pixels + ROWS*COLS*1 + i*COLS + j ) = p[j*3 + 1];
      *( pixels + ROWS*COLS*2 + i*COLS + j ) = p[j*3 + 2];
    }
  }

  cvReleaseImage( &img );
  return true;
}

void show_rectangle(IplImage* img, CvRect& rect , char* window_name, CvScalar scalar, double scale = 1.0){
  IplImage* img_show = cvCreateImage( cvGetSize( img ), 8, 3);
  cvCopy( img, img_show );
  cvRectangle( img_show, cvPoint(rect.x/scale, rect.y/scale), 
    cvPoint( rect.x/scale+rect.width/scale, rect.y/scale+rect.height/scale ),
    scalar );
  cvShowImage( window_name, img_show );
  cvReleaseImage( &img_show );
}

void draw_rectangle(IplImage* img, CvRect& rect , CvScalar scalar, double scale = 1.0){

  cvRectangle( img, cvPoint(rect.x/scale, rect.y/scale), 
    cvPoint( rect.x/scale+rect.width/scale, rect.y/scale+rect.height/scale ), scalar );
}

void SlideWindow( string& path, int cam_num, double threshold, 
  double w_h_ratio,  double scale, int channels, std::vector<Proposal>& result_vec ){
  IplImage* img = NULL;
  if (channels == 1)
    img = cvLoadImage( path.c_str(), 0 );
  else
    img = cvLoadImage( path.c_str() );
  //IplImage* save = cvCreateImage( cvSize(img->width, img->height), 8, 3);
  //cvCopy( img, save );
  CvRect rect;
  if (img == NULL){
    return;
  }
  IplImage* img2 = cvCreateImage( cvSize(img->width*scale, img->height*scale), 8, img->nChannels );
  cvResize( img, img2 );

  //create slide windows
  int width, height, x,y;
  IplImage* img_window = NULL;
  IplImage* dst_img = cvCreateImage( cvSize(COLS, ROWS), 8, img->nChannels );
  int start_pos;
  Proposal one;
  double ave;
  double tmp;
  int count = COLS*ROWS*channels;
  int start_stride, stride;

  stride = 10;
  if (cam_num == 1){
    start_pos = 50*scale;
    start_stride = 10;
  }
  else if (cam_num == 2){
    start_pos = 50*scale;
    start_stride = 10;
  }
  else if (cam_num == 3){
    start_pos = 70*scale;
    start_stride = 10;
  }
  else if (cam_num == 5){
    start_pos = 50*scale;
    start_stride = 10;
  }

  for (y=start_pos; true; y+=stride){//position
    if (cam_num == 1)
      width = height = ( (y-70)/2.875 + 50 );//*scale;
    else if (cam_num == 2)
      width = height = ( (y-60)/3.125 + 20 );//*scale;
    else if (cam_num == 3)
      width = height = ( (y-100)/2.589 + 30 );//*scale;
    else if (cam_num == 5)
      width = height = ( (y-55)/2.579 + 25 );//*scale;
    
    //cvCopy( img, save );
    if ( y+height>=img2->height )
        break;
    img_window = cvCreateImage( cvSize(width, height), 8, img2->nChannels );

    for (x=0; true; x+=stride){
      if ( x+width>=img2->width )
        break;
      rect.x = x; rect.y = y; rect.width = width; rect.height = height;
      cvSetImageROI( img2, rect );
      cvCopy( img2, img_window );
      cvResize( img_window, dst_img );
      //get data
      one.data.reset( new float[ COLS*ROWS*(img2->nChannels) ] );

      ave = 0;
      for (int i=0; i<ROWS; i++){
        unsigned char* p = (unsigned char*)( dst_img->imageData + dst_img->widthStep*i );
        for (int j=0; j<COLS; j++){
          for (int c=0; c<dst_img->nChannels; c++) {
            tmp = (float)p[j*dst_img->nChannels + c]*0.00390625;
            one.data.get()[ ROWS*COLS*c + i*COLS + j ] = tmp;
            ave += tmp;
          }
        }
      }
      ave /= count;
      for (int i=0; i<count; i++){
        one.data.get()[i] -= ave;
      }
      one.x = rect.x/scale;
      one.y = rect.y/scale;
      one.width = rect.width/scale;
      one.height = rect.height/scale;
      result_vec.push_back( one );
    }
    cvReleaseImage( &img_window );
  }
  cvReleaseImage( &img );
  cvReleaseImage( &img2 );
  cvReleaseImage( &dst_img );
}

void SlideWindow( IplImage* img, int cam_num, double threshold, 
  double w_h_ratio, int channels, std::vector<Proposal>& result_vec ){
  // IplImage* img = NULL;
  // if (channels == 1)
  //   img = cvLoadImage( path.c_str(), 0 );
  // else
  //   img = cvLoadImage( path.c_str() );
  //IplImage* save = cvCreateImage( cvSize(img->width, img->height), 8, 3);
  //cvCopy( img, save );
  CvRect rect;
  if (img == NULL){
    return;
  }
  //cout<<"Here1\n";
  IplImage* img2 = cvCreateImage( cvSize(img->width, img->height), 8, img->nChannels );
  cvResize( img, img2 );

  //cout<<"Here2\n";
  //create slide windows
  int width, height, x,y;
  IplImage* img_window = NULL;
  IplImage* dst_img = cvCreateImage( cvSize(COLS, ROWS), 8, img->nChannels );
  int start_pos;
  Proposal one;
  double ave;
  double tmp;
  int count = COLS*ROWS*channels;  
  int start_stride, stride;

  if (cam_num == 1){
    start_pos = 50;
    start_stride = 12;
  }
  else if (cam_num == 2){
    start_pos = 50;
    start_stride = 6;
  }
  else if (cam_num == 3){
    start_pos = 70;
    start_stride = 11;
  }
  else if (cam_num == 5){
    start_pos = 75;
    start_stride = 9;
  }

  for (y=start_pos, stride = start_stride; true; y+=stride){//position
    if (cam_num == 1){
      width = height = ( (y-70)/2.875 + 50 );
      stride = (y-start_pos)*(15-start_stride)/(576-start_pos) + start_stride;
    }
    else if (cam_num == 2){
      width = height = ( (y-60)/3.125 + 20 );
      stride = (y-start_pos)*(15-start_stride)/(576-start_pos) + start_stride;
    }
    else if (cam_num == 3){
      width = height = ( (y-100)/2.589 + 30 );
      stride = (y-start_pos)*(15-start_stride)/(576-start_pos) + start_stride;
    }
    else if (cam_num == 5){
      width = height = ( (y-55)/2.579 + 25 );
      stride = (y-start_pos)*(15-start_stride)/(576-start_pos) + start_stride;
    }
    
    //cvCopy( img, save );
    if ( y+height>=img2->height )
        break;
    img_window = cvCreateImage( cvSize(width, height), 8, img2->nChannels );

    for (x=0; true; x+=stride){
      if ( x+width>=img2->width )
        break;
      rect.x = x; rect.y = y; rect.width = width; rect.height = height;
      cvSetImageROI( img2, rect );
      cvCopy( img2, img_window );
      cvResize( img_window, dst_img );
      //get data
      one.data.reset( new float[ COLS*ROWS*(img2->nChannels) ] );

      ave = 0;
      for (int i=0; i<ROWS; i++){
        unsigned char* p = (unsigned char*)( dst_img->imageData + dst_img->widthStep*i );
        for (int j=0; j<COLS; j++){
          for (int c=0; c<dst_img->nChannels; c++) {
            tmp = (float)p[j*dst_img->nChannels + c]*0.00390625;
            one.data.get()[ ROWS*COLS*c + i*COLS + j ] = tmp;
            ave += tmp;
          }
        }
      }
      ave /= count;
      for (int i=0; i<count; i++){
        one.data.get()[i] -= ave;
      }
      one.x = rect.x;
      one.y = rect.y;
      one.width = rect.width;
      one.height = rect.height;
      result_vec.push_back( one );
    }
    //cout<<"Here4\n";
    cvReleaseImage( &img_window );
  }
  //cout<<"here5\n";
  //cvReleaseImage( &img );
  cvReleaseImage( &img2 );
  cvReleaseImage( &dst_img );
  //cout<<"here6\n";
}
double interUnio(const Proposal &A, const Proposal &B)
{
  int area, temp;

  area = A.width * A.height;
  temp = B.width*B.height;

  if (area > temp)
    area = temp;
  //inter
  int x1 = max( A.x, B.x );
  int x2 = min( A.x+A.width, B.x+B.width );
  int y1 = max( A.y, B.y );
  int y2 = min( A.y+A.height, B.y+B.height );

  if (y1-y1<0 || x2-x1<0)
    temp = 0;
  else
    temp = (y2-y1) * (x2-x1);

  return (double)temp/area;
}

void NonmaximumSupression( std::vector<Proposal>& result ){

  //connect region
  int obj_id = 0;
  int i=0,j=0, size = result.size();
  std::vector<std::vector<Proposal> > connect_region;
  std::vector<Proposal> one_region;
  double inter_unio;
  for (i=0; i<size; i++) {
    result[i].id = -1;
  }
  for (i=0; i<size; i++) {
    //get a connect region
    if (result[i].id == -1) {    
      result[i].id = obj_id;
      obj_id++;
      one_region.clear();
      one_region.push_back( result[i] );
      connect_region.push_back( one_region );
    }
    
    for (j=0; j<size; j++) {
      if (result[j].id != -1)
        continue;
      inter_unio = interUnio(result[i], result[j]);

      if ( inter_unio > 0.7 ) { 
        result[j].id = result[i].id;
        connect_region[ result[j].id ].push_back( result[j] );
      }
    }
  }
  //filt by score
  result.clear();
  size = connect_region.size();
  Proposal one;
  int flag = 0;
  for (i=0; i<size; i++){
    if (connect_region[i].size() == 0)
      continue;
    one = connect_region[i][0];
    one.num = 1;
    for (j=1; j<connect_region[i].size(); j++){
      if (one.score < connect_region[i][j].score){
        one = connect_region[i][j];
        one.num = 1;
      }
      else if (one.score == connect_region[i][j].score) {
        one.x += connect_region[i][j].x;
        one.y += connect_region[i][j].y;
        one.num += 1;
      }
    }
    one.x /= one.num;
    one.y /= one.num;
    result.push_back( one );
  }
}

void NonmaximumSupression2( std::vector<Proposal>& result, const double threshold ){
  std::vector< Proposal > final_result;
  std::vector<Proposal>::iterator it;
  int final_size = 0;
  int idx;
  // 1.sort descending order
  sort(result.begin(), result.end(), greater<Proposal>());
  it = result.begin();
  while( result.empty() != true ) {
    // 2.move the first element of the result detection list to the end
    // of the final detection list
    final_result.push_back( result[0] );
    final_size++;
    result.erase( result.begin() );
    // 3.delete all elements of the result detection list that overlap more
    // than a predetermined threshold with the last elements of the final detection list
    for(idx = result.size()-1; idx>=0; idx-- ){
      if (interUnio( result[idx], final_result[final_size-1] ) > threshold){
        result.erase( result.begin()+idx );
      }
    }
    // 4.go to 2. if the result detection list is not empty
  }
  result.clear();

  result.assign( final_result.begin(), final_result.end() );
}

int CreateDir(const char *sPathName)  
{  
  char DirName[256];  
  strcpy(DirName, sPathName);  
  int i,len = strlen(DirName);  
  if(DirName[len-1]!='/')  
    strcat(DirName, "/");  
   
  len = strlen(DirName);  
   
  for(i=1; i<len; i++) {  
    if(DirName[i]=='/') {  
      DirName[i]   =   0;  
      if( access(DirName, 0) != 0 ) {  
          if(mkdir(DirName, 0755)==-1) {
            perror("mkdir error");
            return -1;   
          }  
      }  
      DirName[i] = '/';  
    }  
  }  
   
  return   0;  
} 

void show_final_result( string &path, std::vector<Proposal>& result, char*name, 
  int cam_num, double threshold ){

  int i=0, size = result.size();
  IplImage* img = cvLoadImage( path.c_str() );
  CvRect rect;
  srand(time(NULL));  

  for (i=0; i<size; i++){
    rect.x = result[i].x;
    rect.y = result[i].y;
    rect.width = result[i].width;
    rect.height = result[i].height;

    //draw_rectangle( img, rect, cvScalar(rand()%256, rand()%256, rand()%256), 1 );
    draw_rectangle( img, rect, 
      cvScalar(0, 0, (result[i].score-threshold)*255/(1-threshold)), 1 );
  }
  char file[100];
  sprintf( file, "result/cam_%d/threshold_%f/", cam_num, threshold );
  if ( CreateDir( file ) == -1 ){
    printf("Create Directory %s failed\n", file);
    return;
  }
  sprintf( file, "result/cam_%d/threshold_%f/%s_%s", 
    cam_num, threshold, path.substr( path.find_last_of('/')+1 ).c_str(), name);
  cvSaveImage( file, img );
  cvReleaseImage( &img );
}
