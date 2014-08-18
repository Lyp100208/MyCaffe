#include "test.h"

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
  if (argc < 8) {
    LOG(ERROR) << "test_SED_HS_net.bin SED_HS_test_my.prototxt ***_quick_iter_** file_list [CPU/GPU] cam_num threshold channels";
    return -1;
  }

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);

  if (argc == 8 && strcmp(argv[4], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
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

  int rows = 32, cols = 32;
  string line;
  int cam_num = atoi( argv[5] );
  double threshold = atof( argv[6] );
  int channels = atoi( argv[7] );

  getline(list_file, line);
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
  while( line != "" && !list_file.eof() ) {
    // caffe_test_net.ReInit( test_net_param, 1 );
    // caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
    cout<<line<<endl;
    result.clear();
    result2.clear();
    // for (int scale_i = 0; scale_i<5; scale_i++){
    //   SlideWindow(line, cam_num, threshold, 1.0, 3, scale_i*0.2+SCALE, channels, result);
    //   cout<<"SCALE = "<<scale_i<<"\tsize = "<<result.size()<<endl;
    // }
    SlideWindow(line, cam_num, threshold, 1.0, 1.0, channels, result);
    //classify
    size = result.size();
    cout<<"size = "<<size<<"\n";
    totaltime = totaltime1 = 0;
    for (i=size; i>0; i-=batch_size){
      if (i >= batch_size){
        deal_size = batch_size;
      }else{
        deal_size = i;
      }
      start = clock();
      for (j=0; j<deal_size; j++){
        memcpy(data+j*COUNT, result[ i-j-1 ].data.get(), sizeof(float)*COUNT);
      }
      finish = clock();
      totaltime += (double)( finish-start )/CLOCKS_PER_SEC;
      start1 = clock();
      const vector<Blob<float>*>& test_result = 
        caffe_test_net.MyForward( data, COUNT*batch_size, label, batch_size );
      finish1 = clock();
      totaltime1 += (double)( finish1-start1 )/CLOCKS_PER_SEC;

      const float* float_data = test_result[0]->cpu_data();
      for (j=0; j<deal_size; j++){
        score = float_data[ (j+1)*3+2 ];
        if ( (int)(float_data[ (j+1)*3 ]) == 1 && score > threshold ){
          result[ i-j-1 ].score = score;
          result2.push_back( result[ i-j-1 ] );
        }
      }
    }

    cout<<"\n此程序的运行时间为"<<totaltime<<"s！"<<totaltime1<<"s"<<endl;

    show_final_result(line, result2, (char*)"final_0.jpg", cam_num, threshold);
    cout<<"result size = "<< result2.size()<<endl;
    NonmaximumSupression2( result2, 0.7 );
    cout<<"after result size = "<< result2.size()<<endl;
    show_final_result(line, result2, (char*)"final_1.jpg", cam_num, threshold);
    NonmaximumSupression2( result2, 0.5 );
    cout<<"after result size = "<< result2.size()<<endl;
    show_final_result(line, result2, (char*)"final_2.jpg", cam_num, threshold);

    getline( list_file, line ); 

  }
  delete []data;
  delete []label;

  return 0;
}
