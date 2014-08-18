#include "test.h"
#include <stdio.h>
class Result{
public:
  int root[4];
  // int sub[32];
  // int which;
  float confidence;
};
void draw_rect( IplImage* img, Result* result ,const float threshold, const int size){

  int i=0;
  srand(time(NULL));  

  for (i=0; i<size; i++){

    //draw_rectangle( img, rect, cvScalar(rand()%256, rand()%256, rand()%256), 1 );
    if (result[i].confidence >= threshold)
      cvRectangle( img, cvPoint(result[i].root[0], result[i].root[1]), 
        cvPoint( result[i].root[2], result[i].root[3] ), cvScalar(0, 0, 255) , 2);
    // draw_rectangle( img, rect, 
    //   cvScalar(0, 0, 255), 1 );
  }
}
int main(int argc, char** argv) {
  //::google::InitGoogleLogging(argv[0]);
  if (argc < 7) {
    cout << "get_feature.bin test_my.prototxt ***_quick_iter_** file_list [CPU/GPU] threshold channels";
    return -1;
  }

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);

  if (argc == 7 && strcmp(argv[4], "GPU") == 0) {
    cout << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    cout << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[1], &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  // Open files
  std::ifstream list_file(argv[3], std::ios::in | std::ios::binary);
  CHECK(list_file) << "Unable to open file " << argv[3];
  cout<<argv[3]<<endl;

  int rows = 32, cols = 32;
  string line, save_name, video_name, detect_name, frameid_path;
  int cam_num;// = atoi( argv[5] );
  double threshold = atof( argv[5] );
  int channels = atoi( argv[6] );

  std::vector<Proposal> proposal;
  Proposal one_proposal;
  float* one_data;
  std::vector<Proposal> result2;
  const double SCALE = 0.4;
  float score;
  int i,j, size, deal_size;
  const int batch_size = test_net_param.layers(0).layer().batchsize();
  const int COUNT = rows*cols*channels;
  float* data = new float[ COUNT * batch_size ];
  float* label = new float[ batch_size ];
  for (int i=0; i<batch_size; i++){
    label[i] = 1.0;
  } 

  clock_t start,finish, start1, finish1;
  double totaltime, totaltime1;
  cout<<"COUNT = "<<COUNT<<endl;

  IplImage *frame;  
  int fid = 0, num, fid_read;
  int count = 0;
  int rtn;
  double ave;
  double tmp;
  Result* result;

  FILE* save_file = NULL;
  FILE* detect_file = NULL;
  CvCapture *capture = NULL;

  CvRect rect;
  IplImage* img_window;
  IplImage* dst_img = cvCreateImage( cvSize(COLS, ROWS), 8, channels );
  int di;
  string tmp_str;
  //循环顺序地读取视频中的帧  
  while( !list_file.eof() ) {
    line = "";
    getline( list_file, line ); 
    if (line == "")
      continue;
    // Create result files
    video_name = line.substr( 0, line.find_last_of(' ') );
    frameid_path = video_name+".frame";
    cam_num = atoi( line.substr( line.find_last_of(' ')+1, line.length()-line.find_last_of(' ') ).c_str() );
    detect_name = video_name.substr( video_name.find_last_of('/')+1, video_name.length()-video_name.find_last_of('/') ); 
    if (detect_name[0] == 'L'){
      save_name = "DetectionResult/train/" + detect_name + ".feature"; 
      detect_name = "DetectionResult/train/" + detect_name + ".data"; 
    }else{
      save_name = "DetectionResult/test/" + detect_name + ".feature"; 
      detect_name = "DetectionResult/test/" + detect_name + ".data"; 
    }
    cout
    <<"\nvideo_name = "<<video_name
    <<"\ncam_num = "<<cam_num
    <<"\ndetect_name = "<<detect_name
    <<"\nframe_path = "<<frameid_path
    <<"\nsave_name = "<<save_name<<endl;

    save_file = fopen(save_name.c_str(), "wb");
    detect_file = fopen(detect_name.c_str(), "rb");
    if (detect_file == NULL){
      cout<<"Can't open detect_file : "<<detect_file<<endl;
      continue;
    }
    ifstream frame_file( frameid_path.c_str(), ios::in );
    if (!frame_file){
      cout<<"Can't open frame file "<<frameid_path<<endl;
      continue;
    }
    capture = cvCreateFileCapture(video_name.c_str()); 
    CHECK(capture) << "Unable to open file " << video_name;
    start = clock();
    fid = 0;
    count = 0;
    rtn = 1;
    frame_file>>tmp_str;
    while( frame_file>>fid_read )  
    {  
      for (;fid<fid_read; fid++){
        frame = cvQueryFrame(capture);  
        if(!frame)  
        {   // 如果没有读取到帧的话，则说明视频播放完毕了，从而退出播放  
            break;  
        }  
      }
      if(!frame)  
      {   // 如果没有读取到帧的话，则说明视频播放完毕了，从而退出播放  
        break;  
      }  
      count++;
      if (fid < count){
        rtn = fread( (char*)&fid, sizeof(int), 1, detect_file );
        rtn = fread( (char*)&num, sizeof(int), 1, detect_file );
      }
      if (num == 0)
        continue;
      if (count%200 == 0)
        cout<<count<<" "<<flush;
      //cout<<"num = "<<num<<endl;
      if (fid != count){
        continue;
      }
      result = new Result[num];
      rtn = fread( (char*)result, sizeof(Result), num, detect_file );
      //if (result.confidence >= threshold)
      //draw_rect( frame, result, threshold, num );

      // cvShowImage( "Result", frame );
      // cvWaitKey(  );
      //get proposal
      proposal.clear();
      for (di=0; di<num; di++) {
        rect.x = result[di].root[0]; rect.y = result[di].root[1]; 
        rect.width = abs( result[di].root[2]-result[di].root[0] ); 
        rect.height = abs( result[di].root[3]-result[di].root[1] ); 
        // cout<<result[di].root[0]<<","<<result[di].root[1]<<","<<result[di].root[2]<<","<<result[di].root[3]<<endl;
        // cout<<"x = "<<rect.x<<"y = "<<rect.y<<"w = "<<rect.width<<"h = "<<rect.height<<endl;
        img_window = cvCreateImage( cvSize(rect.width, rect.height), 8, frame->nChannels );
        cvSetImageROI( frame, rect );
        cvCopy( frame, img_window );
        cvResize( img_window, dst_img );
        cvReleaseImage( &img_window );
        //get data
        one_data = new float[ COLS*ROWS*(frame->nChannels) ];
        one_proposal.data.reset( one_data );

        ave = 0;
        for (int i=0; i<ROWS; i++){
          unsigned char* p = (unsigned char*)( dst_img->imageData + dst_img->widthStep*i );
          for (int j=0; j<COLS; j++){
            for (int c=0; c<dst_img->nChannels; c++) {
              tmp = (float)p[j*dst_img->nChannels + c]*0.00390625;
              one_proposal.data.get()[ ROWS*COLS*c + i*COLS + j ] = tmp;
              ave += tmp;
            }
          }
        }
        ave /= COUNT;
        for (int i=0; i<COUNT; i++){
          one_proposal.data.get()[i] -= ave;
        }
        one_proposal.x = rect.x;
        one_proposal.y = rect.y;
        one_proposal.width = rect.width;
        one_proposal.height = rect.height;
        one_proposal.score = result[di].confidence;
        proposal.push_back( one_proposal );
      }

      //get feature
      size = proposal.size();
      totaltime = totaltime1 = 0;
      for (i=size; i>0; i-=batch_size){
        if (i >= batch_size){
          deal_size = batch_size;
        }else{
          deal_size = i;
        }
        //start = clock();
        for (j=0; j<deal_size; j++){
          memcpy(data+j*COUNT, proposal[ i-j-1 ].data.get(), sizeof(float)*COUNT);
        }
        // finish = clock();
        // totaltime += (double)( finish-start )/CLOCKS_PER_SEC;
        // start1 = clock();
        const vector<Blob<float>*>& feature = 
          caffe_test_net.MyForward_GetFeature( data, COUNT*batch_size, label, batch_size, "ip5");
        //finish1 = clock();
        //totaltime1 += (double)( finish1-start1 )/CLOCKS_PER_SEC;

        const float* float_data = feature[0]->cpu_data();
        for (j=0; j<deal_size; j++){
          fwrite( &(result[i-j-1]), sizeof( Result ), 1, save_file );
          fwrite( float_data+j*64, sizeof(float), 64, save_file );
        }
      }
      delete [] result;
    }
    fclose( save_file );
    fclose( detect_file );
    cvReleaseCapture( &capture );

    finish = clock();
    totaltime = (double)( finish-start )/CLOCKS_PER_SEC;
    cout<<"\n此程序的运行时间为"<<totaltime<<"s！"<<totaltime1<<"s"<<endl;
  }
  cvReleaseImage( &dst_img );
  delete []data;
  delete []label;

  return 0;
}
