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

void weight_to_picture(shared_ptr<Blob<float> >& weight_blob, 
	std::vector< IplImage* >& pictures)
{
	int num = weight_blob->num();
	int channels = weight_blob->channels();
	int height = weight_blob->height();
	int width = weight_blob->width();

	cout<<"num = "<<num<<endl
	<<"channels = "<<channels<<endl
	<<"height = "<<height<<endl
	<<"width = "<<width<<endl;

	float max = -999999999, min = 9999999999, tmp;
	unsigned char* p;
	int i,c,row, col;
	for ( i=0; i<num; i++) {
		for ( c=0; c<channels; c++){
			for ( row=0; row<height; row++){
				for (col=0; col<width; col++){
					tmp = weight_blob->data_at(i, c, row, col);
					if (tmp > max)
						max = tmp;
					if (tmp < min)
						min = tmp;
				}
			}
			//normalize
			IplImage* img = cvCreateImage( cvSize(height, width), 8, 1 );
			for (row=0; row<height; row++){
				p = (unsigned char*)( img->imageData + img->widthStep*row );
				for (col=0; col<width; col++){
					p[col] = ( weight_blob->data_at(i, c, row, col) - min ) * 255
					/ ( max - min );
				}
			}
			pictures.push_back( img );
		}
	}
}

void show_weight( const vector<shared_ptr<Layer<float> > >& layers )
{
	int size = layers.size();
	std::vector< IplImage* > pictures;
	char path[100];

	for (int i=0; i<size; i++){
		const LayerParameter& param = layers[i]->layer_param();
		cout<<"layer "<<i<<" : "<<param.type()
			<<"\t"<<param.name()<<endl;
		if (param.type() == "conv"){
			pictures.clear();
			weight_to_picture( layers[i]->blobs()[0], pictures );
			for (int j=0; j<pictures.size(); j++){
				IplImage* src = cvCreateImage( cvSize( pictures[j]->width*12, pictures[j]->height*12 ),
					8, 1);
				cvResize( pictures[j], src );
				sprintf(path, "WeightImage/%s_map%d.jpg", param.name().c_str(), j);
				cvSaveImage( path, src );
				// cout<<"weight : "<<j<<endl;
				// cvShowImage( "weight", pictures[j] );
				// cvWaitKey( 0 );
				cvReleaseImage( &pictures[j] );
				cvReleaseImage( &src );
			}
		} 
	}
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
  if (argc < 3) {
    LOG(ERROR) << "show_weight.bin ***_test.prototxt ***_quick_iter_**";
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
  const vector<shared_ptr<Layer<float> > >& layers = 
  	caffe_test_net.layers();
  show_weight( layers );
  return 0;
}