#include "test.h"
#include <stdio.h>

int main(int argc, char** argv) {
	//::google::InitGoogleLogging(argv[0]);
  if (argc < 7) {
    cout << "test_video.bin test_my.prototxt ***_quick_iter_** file_list [CPU/GPU] threshold channels";
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
  string line, save_name, video_name;
  int cam_num;// = atoi( argv[5] );
  double threshold = atof( argv[5] );
  int channels = atoi( argv[6] );

  std::vector<Proposal> result;
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
  CreateDir("DetectionResult");
  FILE* save_file = NULL;
  CvCapture *capture = NULL;
  int fid;
  //循环顺序地读取视频中的帧  
  while( !list_file.eof() ) {

    getline( list_file, line ); 
    if (line == "")
      continue;
    // Create result files
    video_name = line.substr( 0, line.find_last_of(' ') );
    cam_num = atoi( line.substr( line.find_last_of(' ')+1, line.length()-line.find_last_of(' ') ).c_str() );
    save_name = video_name.substr( video_name.find_last_of('/')+1, video_name.length()-video_name.find_last_of('/') ); 
    save_name = "DetectionResult/" + save_name + ".data"; 
    cout
    <<"\nvideo_name = "<<video_name
    <<"\ncam_num = "<<cam_num
    <<"\nsave_name = "<<save_name<<endl;

    save_file = fopen(save_name.c_str(), "wb");
    capture = cvCreateFileCapture(video_name.c_str()); 
    CHECK(capture) << "Unable to open file " << video_name;
    start = clock();
    fid = 0;
    while(1)  
    {  
      // 获取当前播放帧的下一帧，并且将获取到的帧加载到内存中，覆盖掉前面帧所占的内存  
      frame = cvQueryFrame(capture);  
      if(!frame)  
      {   // 如果没有读取到帧的话，则说明视频播放完毕了，从而退出播放  
          break;  
      }  
      fid++;
      result.clear();
      result2.clear();
      SlideWindow(frame, cam_num, threshold, 1.0, channels, result);
      //cvShowImage("abc", frame);
      //cvWaitKey(0);
      //classify
      size = result.size();
      totaltime = totaltime1 = 0;
      for (i=size; i>0; i-=batch_size){
        if (i >= batch_size){
          deal_size = batch_size;
        }else{
          deal_size = i;
        }
        //start = clock();
        for (j=0; j<deal_size; j++){
          memcpy(data+j*COUNT, result[ i-j-1 ].data.get(), sizeof(float)*COUNT);
        }
        // finish = clock();
        // totaltime += (double)( finish-start )/CLOCKS_PER_SEC;
        // start1 = clock();
        const vector<Blob<float>*>& test_result = 
          caffe_test_net.MyForward( data, COUNT*batch_size, label, batch_size );
        //finish1 = clock();
        //totaltime1 += (double)( finish1-start1 )/CLOCKS_PER_SEC;

        const float* float_data = test_result[0]->cpu_data();
        for (j=0; j<deal_size; j++){
          score = float_data[ (j+1)*3+2 ];
          if ( (int)(float_data[ (j+1)*3 ]) == 1 && score > threshold ){
            result[ i-j-1 ].score = score;
            result2.push_back( result[ i-j-1 ] );
          }
        }
      }

      // show_final_result(line, result2, (char*)"final_0.jpg", cam_num, threshold);
      // cout<<"result size = "<< result2.size()<<endl;
      //NonmaximumSupression2( result2, 0.7 );
      // cout<<"after result size = "<< result2.size()<<endl;
      // show_final_result(line, result2, (char*)"final_1.jpg", cam_num, threshold);
      NonmaximumSupression2( result2, 0.5 );
      //cout<<"result size = "<< result2.size();
      // show_final_result(line, result2, (char*)"final_2.jpg", cam_num, threshold);
      if (fid%100 == 0){
        cout<<"fid="<<fid<<" "<<flush;
      }
      int i=0, rsize = result2.size(), val;
      fwrite(&fid, sizeof(int), 1, save_file);
      fwrite(&rsize, sizeof(int), 1, save_file);
      for (i=0; i<rsize; i++){
        fwrite(&(result2[i].x), sizeof(int), 1, save_file);
        fwrite(&(result2[i].y), sizeof(int), 1, save_file);
        val = result2[i].x+result2[i].width;
        fwrite(&val, sizeof(int), 1, save_file);
        val = result2[i].y+result2[i].height;
        fwrite(&val, sizeof(int), 1, save_file);
        fwrite(&(result2[i].score), sizeof(float), 1, save_file);
      }
    }
    finish = clock();
    totaltime = (double)( finish-start )/CLOCKS_PER_SEC;
    cout<<"\n此程序的运行时间为"<<totaltime<<"s！"<<totaltime1<<"s"<<endl;
  }
  delete []data;
  delete []label;
  cvReleaseCapture( &capture );
  fclose( save_file );

  return 0;
}
