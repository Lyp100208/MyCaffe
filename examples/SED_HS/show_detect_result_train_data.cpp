#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;
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
int main(int argc, char** argv)
{
	if(argc<4){
		cout<<"Usage: ./show_detect_result result_file video_folder threshold"<<endl;
		return -1;
	}
	string result_file = argv[1];
	string video_folder = argv[2];
	string video_name,  str;
	char video_path[100];
	//int cam_num;
	float threshold = atof( argv[3] );

	video_name = result_file.substr( result_file.find_last_of('/')+1, result_file.length()-result_file.find_last_of('/') ); 
	video_name = video_name.substr( 0, video_name.length()-5 );
	// str = video_name.substr( 7, 2 );
	// cam_num = atoi( str.c_str() );
	sprintf(video_path, "%s/%s", video_folder.c_str(), video_name.c_str());

	CvCapture* capture = cvCreateFileCapture(video_path); 
	if (!capture){
    	cout << "Unable to open file " << video_path << endl;
    	return -1;
	}
    FILE* fptr = fopen(result_file.c_str(), "rb");
    if (!fptr){
    	cout<<"Unable to open file "<<result_file<<endl;
    	return -1;
    }

    int fid = 0, num;
    int count = 0;
    int rtn;
    Result* result;
    IplImage* image;

    while( !feof( fptr ) )
    {
		image = cvQueryFrame(capture);  
		if(!image)  
		{   // 如果没有读取到帧的话，则说明视频播放完毕了，从而退出播放  
		  break;  
		}  
		count++;
		if (fid < count){
	    	rtn = fread( (char*)&fid, sizeof(int), 1, fptr );
			rtn = fread( (char*)&num, sizeof(int), 1, fptr );
		}

		if (fid == count && num > 0){
			result = new Result[num];
			rtn = fread( (char*)result, sizeof(Result), num, fptr );
			//if (result.confidence >= threshold)
			draw_rect( image, result, threshold, num );
			delete [] result;
		}
		cvShowImage( "Result", image );
		cvWaitKey( 40 );
    }
    fclose( fptr );
    cvReleaseCapture( &capture );
	return 0;
}