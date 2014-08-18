#include "test.h"
#include <dirent.h>  
#include <sys/stat.h>  
#include <unistd.h>  
#include <sys/types.h> 
#include <fstream>
using namespace std;

class Result{
public:
  Result(){
    root[0] = root[1] = root[2] = root[3] = -1;
  }
  int root[4];
  // int sub[32];
  // int which;
  float confidence;
};

void listDir(char *path, vector<string>& path_vec)  
{  
  DIR              *pDir ;  
  struct dirent    *ent  ;  
  int               i=0  ;  
  char              childpath[512];  

  pDir=opendir(path);  
  memset(childpath,0,sizeof(childpath));  

  while((ent=readdir(pDir))!=NULL)  
  {  

    if(ent->d_type & DT_DIR)  
    {  
      if(strcmp(ent->d_name,".")==0 || strcmp(ent->d_name,"..")==0)  
        continue;  
      sprintf(childpath,"%s/%s",path,ent->d_name);  
      listDir(childpath, path_vec);  

    }  else {
			sprintf( childpath, "%s/%s", path, ent->d_name );
			if ( childpath[ strlen( childpath ) -1 ] == 'a' )
				path_vec.push_back( childpath );
   	}
  }  
}  
void draw_rect( IplImage* img, Result* result ,const float threshold, const int size){
  int i=0;
  srand(time(NULL));  
  if (result == NULL)
    return;
  for (i=0; i<size; i++){

    //draw_rectangle( img, rect, cvScalar(rand()%256, rand()%256, rand()%256), 1 );
    if (result[i].confidence >= threshold)
      cvRectangle( img, cvPoint(result[i].root[0], result[i].root[1]), 
      cvPoint( result[i].root[2], result[i].root[3] ), cvScalar(0, 0, 255) , 2);
    // draw_rectangle( img, rect, 
    //   cvScalar(0, 0, 255), 1 );
  }
}

void get_proposal( IplImage* img, Proposal& a_proposal, Result& detect, int fid, const float* mean, int flip = 0 )
{
  CvRect rect;
  rect.x = detect.root[0];
  rect.y = detect.root[1];
  rect.width = detect.root[2] - rect.x;
  rect.height = detect.root[3] - rect.y;
  IplImage* img_window = cvCreateImage( cvSize(rect.width, rect.height), 8, img->nChannels );
  IplImage* dst_img = cvCreateImage( cvSize( COLS, ROWS ), 8, img->nChannels );
  int count = COLS*ROWS*(img->nChannels);

  cvSetImageROI( img, rect );
  cvCopy( img, img_window );
  if (flip == 1)
    cvFlip( img_window, NULL, 1 );//沿y轴翻转
  cvResize( img_window, dst_img );

  float* data_f = new float[ COLS*ROWS*(img->nChannels) ]; 
  a_proposal.data.reset( data_f );

  int i,j,c;
  unsigned char*p;
  for (i=0; i<ROWS; i++){
    p = (unsigned char*)( dst_img->imageData + dst_img->widthStep*i );
    for (j=0; j<COLS; j++){
      for (c=0; c<dst_img->nChannels; c++) {
        a_proposal.data.get()[ ROWS*COLS*c + i*COLS + j ] = 
          (float)( p[j*dst_img->nChannels + c]- mean[ ROWS*COLS*c + i*COLS + j ] )*0.00390625;
      }
    }
  }
  a_proposal.x = rect.x;
  a_proposal.y = rect.y;
  a_proposal.width = rect.width;
  a_proposal.height = rect.height;
  a_proposal.score = detect.confidence;
  a_proposal.id = fid;

  cvReleaseImage( &img_window );
  cvReleaseImage( &dst_img );
}
///home/chenqi/workspace/caffe-master/examples/SED_HS/DetectionResult/test
///media/chenqi/BCB265B6B26575B4/TRECVID_VIDEO
int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
  if (argc < 6) {
    LOG(ERROR) << "test_SED_HS_net.bin SED_HS_test_my.prototxt ***_quick_iter_** folder video_folder save_folder";
    return -1;
  }

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);

  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[1], &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
  const float* mean = caffe_test_net.layers()[0]->get_mean();
  for (int i=0; i<3072; i++){
    cout<<mean[i]<<"\t";
  }
  cout<<endl;

  vector<string> path_vec;
  listDir( argv[3], path_vec);
  string video_folder( argv[4] );
  string save_folder( argv[5] );
  string video_path;
  string save_path;
  string file_name;
  char tmp_str[100];
  int rows = 32, cols = 32;
  string line;
  int cam_num;
  double threshold;
  int channels = 3;
  int fid, box_size,b_i, x1,y1, x2, y2;
  Result* detect_result = NULL;
  ifstream ifs;
  ofstream ofs;
  CvCapture *capture = NULL;
  IplImage* frame = NULL;

  float score;
  int i,j, size, deal_size;
  const int batch_size = test_net_param.layers(0).layer().batchsize();
  const int COUNT = rows*cols*channels;
  float* data = new float[ COUNT * batch_size ];
  float* label = new float[ batch_size ];
  for (int i=0; i<batch_size; i++){
    label[i] = 1.0;
  }

  cout<<"COUNT = "<<COUNT<<endl;
  cout<<"File Num = "<<path_vec.size()<<endl;

  for (int p_i=0; p_i<path_vec.size(); p_i++) {
    cout<<"file = "<<path_vec[p_i]<<endl;
    file_name = path_vec[ p_i ].substr( path_vec[p_i].rfind("/")+1 );
    save_path = save_folder + "/" + file_name + "_point";
    cam_num = file_name[19] - '0';
    file_name = file_name.substr( 0, file_name.length()-5 );
    sprintf( tmp_str, "%s/%s", video_folder.c_str(), file_name.c_str());
    video_path = tmp_str;

    if (cam_num == 2){
      threshold = 0.93;
    }else{
      threshold = 0.96;
    }

    ifs.open( path_vec[ p_i ].c_str(), ios::binary );
    if (ifs.is_open() == false){
      cout<<"Can't open file : "<<path_vec[ p_i ]<<endl;
      continue;
    }
    capture = cvCreateFileCapture(video_path.c_str()); 
    if (capture == NULL){
      cout<<"Can't open video : "<<video_path<<endl;
      continue;
    }
    ofs.open(save_path.c_str(), ios::binary);

    cout<<"video = "<<video_path<<endl;
    cout<<"save_path = "<<save_folder<<endl;
    cout<<"cam_num = "<<cam_num<<endl;
 
    clock_t start,finish;
    int count = 0;
    fid = -1;

    start = clock();

    std::vector<Proposal> proposal_vec;
    std::map< int, vector<Proposal> > point_result_map;
    Proposal a_proposal;
    Result result;
    int w,h;
    IplImage* gray = NULL;

    proposal_vec.clear();
    point_result_map.clear();
    while ( !ifs.eof() ) {
      frame = cvQueryFrame(capture);  
      if( !frame ) {   // 如果没有读取到帧的话，则说明视频播放完毕了，从而退出播放  
        break;  
      }
      if ( channels == 1 &&  gray == NULL){
        gray = cvCreateImage( cvGetSize(frame), frame->depth, 1 );
      }
      if (channels == 1){
        cvCvtColor( frame, gray, CV_RGB2GRAY );
        frame = gray;
      }
      count++;
      if (fid < count){
        ifs.read( (char*)&fid, sizeof(int) );
        ifs.read( (char*)&box_size, sizeof(int) );
      }

      if (fid == count && box_size > 0){
        detect_result = new Result[box_size];
        ifs.read( (char*)detect_result, sizeof(Result)*box_size );

        for (i=0; i<box_size; i++) {
          // if (detect_result[i].confidence < threshold)
          //   continue;
          w = abs( detect_result[i].root[2] - detect_result[i].root[0] )*1.3;
          h = abs( detect_result[i].root[3] - detect_result[i].root[1] )*1.3;
          result.root[0] = detect_result[i].root[0] - w*0.5*(1.0-1.0/1.3);
          result.root[1] = detect_result[i].root[1] - h*0.5*(1.0-1.0/1.3);
          result.root[2] = result.root[0] + w;
          result.root[3] = result.root[1] + h;
          if ( !(result.root[0] < 0 || result.root[2]>= frame->width ||
            result.root[1] < 0 || result.root[3] >= frame->height ||
            w == 0 || h==0) ){
            get_proposal( frame, a_proposal, result, fid, mean );
            proposal_vec.push_back( a_proposal );
          }
        }

        //draw_rect( frame, detect_result, threshold, box_size );
        delete [] detect_result;
      }
      // cvResetImageROI( frame );
      // cvShowImage( "Result", frame );
      // cvWaitKey( 40 );

      size = proposal_vec.size();
      if (size >= 100*batch_size){
        for (i=size; i>0; i-=batch_size){
          if (i >= batch_size){
            deal_size = batch_size;
          }else{
            deal_size = i;
          }
          for (j=0; j<deal_size; j++){
            memcpy(data+j*COUNT, proposal_vec[ i-j-1 ].data.get(), sizeof(float)*COUNT);
          }
          const vector<Blob<float>*>& test_result = 
            caffe_test_net.MyForward( data, COUNT*batch_size, label, batch_size );

          const float* float_data = test_result[0]->cpu_data();
          for (j=0; j<deal_size; j++){
            score = float_data[ (j+1)*3+2 ];
            if ( (int)(float_data[ (j+1)*3 ]) == 1 ){
              proposal_vec[ i-j-1 ].score = score;
              point_result_map[ proposal_vec[i-j-1].id ].push_back( proposal_vec[ i-j-1 ] );
            }
          }
        }
        //cout<<size<<" : "<<point_result_map.size()<<endl;
        proposal_vec.clear();
      }
    }
    size = proposal_vec.size();
    for (i=size; i>0; i-=batch_size){
      if (i >= batch_size){
        deal_size = batch_size;
      }else{
        deal_size = i;
      }
      for (j=0; j<deal_size; j++){
        memcpy(data+j*COUNT, proposal_vec[ i-j-1 ].data.get(), sizeof(float)*COUNT);
      }
      const vector<Blob<float>*>& test_result = 
        caffe_test_net.MyForward( data, COUNT*batch_size, label, batch_size );
      const float* float_data = test_result[0]->cpu_data();
      for (j=0; j<deal_size; j++){
        score = float_data[ (j+1)*3+2 ];
        if ( (int)(float_data[ (j+1)*3 ]) == 1 ){
          proposal_vec[ i-j-1 ].score = score;
          point_result_map[ proposal_vec[i-j-1].id ].push_back( proposal_vec[ i-j-1 ] );
        }
      }
    }
    proposal_vec.clear();
    // save
    map< int, vector<Proposal> >::iterator map_it;
    int total_num = 0, val;
    for (map_it=point_result_map.begin(); map_it!=point_result_map.end(); map_it++) {
      vector<Proposal>& tmp_vec = map_it->second;
      fid = map_it->first;
      box_size = tmp_vec.size();
      total_num += box_size;
      ofs.write( (char*)&fid, sizeof(int) );
      ofs.write( (char*)&box_size, sizeof(int) );
      for (i=0; i<box_size; i++) {
        ofs.write( (char*)&(tmp_vec[i].x), sizeof(int) );
        ofs.write( (char*)&(tmp_vec[i].y), sizeof(int) );
        val = tmp_vec[i].x+tmp_vec[i].width;
        ofs.write( (char*)&val, sizeof(int) );
        val = tmp_vec[i].y+tmp_vec[i].height;
        ofs.write( (char*)&val, sizeof(int) );
        ofs.write( (char*)&(tmp_vec[i].score), sizeof(float) );
      }
    }
    finish = clock();
    cout<<"Total number = "<<total_num<<endl;
    cout<<"运行时间为"<<(double)( finish-start )/CLOCKS_PER_SEC<<"s！"<<endl;
    ofs.close();
    ifs.close();
    cvReleaseCapture( &capture );
    capture = NULL;
  }
  delete []data;
  delete []label;

  return 0;
}


/*
#include "test.h"
#include <dirent.h>  
#include <sys/stat.h>  
#include <unistd.h>  
#include <sys/types.h> 
#include <fstream>
using namespace std;

class Result{
public:
  Result(){
    root[0] = root[1] = root[2] = root[3] = -1;
  }
  int root[4];
  // int sub[32];
  // int which;
  float confidence;
};

void listDir(char *path, vector<string>& path_vec)  
{  
  DIR              *pDir ;  
  struct dirent    *ent  ;  
  int               i=0  ;  
  char              childpath[512];  

  pDir=opendir(path);  
  memset(childpath,0,sizeof(childpath));  

  while((ent=readdir(pDir))!=NULL)  
  {  

    if(ent->d_type & DT_DIR)  
    {  
      if(strcmp(ent->d_name,".")==0 || strcmp(ent->d_name,"..")==0)  
        continue;  
      sprintf(childpath,"%s/%s",path,ent->d_name);  
      listDir(childpath, path_vec);  

    }  else {
			sprintf( childpath, "%s/%s", path, ent->d_name );
			if ( childpath[ strlen( childpath ) -1 ] == 'a' )
				path_vec.push_back( childpath );
   	}
  }  
}  
void draw_rect( IplImage* img, Result* result ,const float threshold, const int size){
  int i=0;
  srand(time(NULL));  
  if (result == NULL)
    return;
  for (i=0; i<size; i++){

    //draw_rectangle( img, rect, cvScalar(rand()%256, rand()%256, rand()%256), 1 );
    if (result[i].confidence >= threshold)
      cvRectangle( img, cvPoint(result[i].root[0], result[i].root[1]), 
      cvPoint( result[i].root[2], result[i].root[3] ), cvScalar(0, 0, 255) , 2);
    // draw_rectangle( img, rect, 
    //   cvScalar(0, 0, 255), 1 );
  }
}

void get_proposal( IplImage* img, Proposal& a_proposal, Result& detect, int fid, int flip = 0 )
{
  CvRect rect;
  rect.x = detect.root[0];
  rect.y = detect.root[1];
  rect.width = detect.root[2] - rect.x;
  rect.height = detect.root[3] - rect.y;
  IplImage* img_window = cvCreateImage( cvSize(rect.width, rect.height), 8, img->nChannels );
  IplImage* dst_img = cvCreateImage( cvSize( COLS, ROWS ), 8, img->nChannels );
  int count = COLS*ROWS*(img->nChannels);

  cvSetImageROI( img, rect );
  cvCopy( img, img_window );
  if (flip == 1)
    cvFlip( img_window, NULL, 1 );//沿y轴翻转
  cvResize( img_window, dst_img );

  float* data_f = new float[ COLS*ROWS*(img->nChannels) ]; 
  a_proposal.data.reset( data_f );

 
  double ave = 0;
  double tmp;
  int i,j,c;
  unsigned char*p;
  for (i=0; i<ROWS; i++){
    p = (unsigned char*)( dst_img->imageData + dst_img->widthStep*i );
    for (j=0; j<COLS; j++){
      for (c=0; c<dst_img->nChannels; c++) {
        tmp = (float)p[j*dst_img->nChannels + c]*0.00390625;
        a_proposal.data.get()[ ROWS*COLS*c + i*COLS + j ] = tmp;
        ave += tmp;
      }
    }
  }
  ave /= count;
  for (i=0; i<count; i++){
    a_proposal.data.get()[i] -= ave;
  }
  a_proposal.x = rect.x;
  a_proposal.y = rect.y;
  a_proposal.width = rect.width;
  a_proposal.height = rect.height;
  a_proposal.score = detect.confidence;
  a_proposal.id = fid;

  cvReleaseImage( &img_window );
  cvReleaseImage( &dst_img );
}
///home/chenqi/workspace/caffe-master/examples/SED_HS/DetectionResult/test
///media/chenqi/BCB265B6B26575B4/TRECVID_VIDEO
int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
  if (argc < 6) {
    LOG(ERROR) << "test_SED_HS_net.bin SED_HS_test_my.prototxt ***_quick_iter_** folder video_folder save_folder";
    return -1;
  }

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);

  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[1], &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  vector<string> path_vec;
  listDir( argv[3], path_vec);
  string video_folder( argv[4] );
  string save_folder( argv[5] );
  string video_path;
  string save_path, feature_path;
  string file_name;
  char tmp_str[100];
  int rows = 32, cols = 32;
  string line;
  int cam_num;
  double threshold;
  int channels = 3;
  int fid, box_size,b_i, x1,y1, x2, y2;
  Result* detect_result = NULL;
  ifstream ifs;
  ofstream ofs, ofs2;
  CvCapture *capture = NULL;
  IplImage* frame = NULL;

  float score;
  int i,j, size, deal_size;
  const int batch_size = test_net_param.layers(0).layer().batchsize();
  const int COUNT = rows*cols*channels;
  float* data = new float[ COUNT * batch_size ];
  float* label = new float[ batch_size ];
  for (int i=0; i<batch_size; i++){
    label[i] = 1.0;
  }

  cout<<"COUNT = "<<COUNT<<endl;

  for (int p_i=0; p_i<path_vec.size(); p_i++) {
    file_name = path_vec[ p_i ].substr( path_vec[p_i].rfind("/")+1 );
    save_path = save_folder + "/" + file_name + "_point";
    cam_num = file_name[8] - '0';
    file_name = file_name.substr( 0, file_name.length()-5 );
    sprintf( tmp_str, "%s/CAM%d/%s", video_folder.c_str(), cam_num, file_name.c_str());
    video_path = tmp_str;
    //feature_path = save_folder + "/" + file_name + ".feature";

    if (cam_num == 2){
      threshold = 0.93;
    }else{
      threshold = 0.96;
    }

    ifs.open( path_vec[ p_i ].c_str(), ios::binary );
    if (ifs.is_open() == false){
      cout<<"Can't open file : "<<path_vec[ p_i ]<<endl;
      continue;
    }
    capture = cvCreateFileCapture(video_path.c_str()); 
    if (capture == NULL){
      cout<<"Can't open video : "<<video_path<<endl;
      continue;
    }
    ofs.open(save_path.c_str(), ios::binary);
    //ofs2.open( feature_path, ios::binary );

    cout<<"file = "<<path_vec[p_i]<<endl;
    cout<<"video = "<<video_path<<endl;
    cout<<"save_path = "<<save_folder<<endl;
    cout<<"cam_num = "<<cam_num<<endl;
 
    clock_t start,finish;
    int count = 0;
    fid = -1;

    start = clock();

    Proposal a_proposal;
    Result result;
    int w,h, val;
    IplImage* gray = NULL;

    while ( !ifs.eof() ) {
      frame = cvQueryFrame(capture);  
      if( !frame ) {   // 如果没有读取到帧的话，则说明视频播放完毕了，从而退出播放  
        break;  
      }
      if ( channels == 1 &&  gray == NULL){
        gray = cvCreateImage( cvGetSize(frame), frame->depth, 1 );
      }
      if (channels == 1){
        cvCvtColor( frame, gray, CV_RGB2GRAY );
        frame = gray;
      }
      count++;
      if (fid < count){
        ifs.read( (char*)&fid, sizeof(int) );
        ifs.read( (char*)&box_size, sizeof(int) );

        ofs.write( (char*)&fid, sizeof(int) );
        ofs.write( (char*)&box_size, sizeof(int) );
      }
      if (count%1000 == 0)
        cout<<count<<" "<<flush;
      if (fid != count || box_size == 0)
        continue;

      detect_result = new Result[box_size];
      ifs.read( (char*)detect_result, sizeof(Result)*box_size );

      std::vector<Proposal> proposal_vec;
      proposal_vec.clear();
      for (i=0; i<box_size; i++) {
        // if (detect_result[i].confidence < threshold)
        //   continue;
        w = abs( detect_result[i].root[2] - detect_result[i].root[0] )*1.5;
        h = abs( detect_result[i].root[3] - detect_result[i].root[1] )*1.5;
        result.root[0] = detect_result[i].root[0] - w/3;
        result.root[1] = detect_result[i].root[1] - h/12;
        result.root[2] = result.root[0] + w;
        result.root[3] = result.root[1] + h;
        if ( !(result.root[0] < 0 || result.root[2]>= frame->width ||
          result.root[1] < 0 || result.root[3] >= frame->height ||
          w == 0 || h==0) ){
          get_proposal( frame, a_proposal, result, fid );
          proposal_vec.push_back( a_proposal );
        }
        result.root[0] = detect_result[i].root[0];
        result.root[1] = detect_result[i].root[1] - h/12;
        result.root[2] = result.root[0] + w;
        result.root[3] = result.root[1] + h;
        if ( !(result.root[0] < 0 || result.root[2]>= frame->width ||
          result.root[1] < 0 || result.root[3] >= frame->height ||
          w == 0 || h==0) ){
          get_proposal( frame, a_proposal, result, fid, 1 );
          proposal_vec.push_back( a_proposal );
        }
      }

      delete [] detect_result;

      size = proposal_vec.size();
      for (i=size; i>0; i-=batch_size){
        if (i >= batch_size){
          deal_size = batch_size;
        }else{
          deal_size = i;
        }
        for (j=0; j<deal_size; j++){
          memcpy(data+j*COUNT, proposal_vec[ i-j-1 ].data.get(), sizeof(float)*COUNT);
        }
        const vector<Blob<float>*>& test_result = 
          caffe_test_net.MyForward( data, COUNT*batch_size, label, batch_size );

        const float* float_data = test_result[0]->cpu_data();
        for (j=0; j<deal_size; j++){
          score = float_data[ (j+1)*3+2 ];
          if ( (int)(float_data[ (j+1)*3 ]) == 1 ){
            proposal_vec[ i-j-1 ].score = score;

            ofs.write( (char*)&(proposal_vec[ i-j-1 ].x), sizeof(int) );
            ofs.write( (char*)&(proposal_vec[ i-j-1 ].y), sizeof(int) );
            val = proposal_vec[ i-j-1 ].x+proposal_vec[ i-j-1 ].width;
            ofs.write( (char*)&val, sizeof(int) );
            val = proposal_vec[ i-j-1 ].y+proposal_vec[ i-j-1 ].height;
            ofs.write( (char*)&val, sizeof(int) );
            ofs.write( (char*)&(proposal_vec[ i-j-1 ].score), sizeof(float) );
          }
        }
      }

      proposal_vec.clear();
    }//end of while

    finish = clock();
    cout<<"运行时间为"<<(double)( finish-start )/CLOCKS_PER_SEC<<"s！"<<endl;
    ofs.close();
    ifs.close();
    cvReleaseCapture( &capture );
    capture = NULL;
  }
  delete []data;
  delete []label;

  return 0;
}



*/