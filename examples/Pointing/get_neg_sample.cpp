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
int count_name = 0;
int cam_num = 0;
void save_neg_image( IplImage* img, Result* result ,const int size){
  int i=0;
  char path[100];

  for (i=0; i<size; i++){

	  CvRect rect;
	  rect.x = result[i].root[0];
	  rect.y = result[i].root[1];
	  rect.width = result[i].root[2] - rect.x;
	  rect.height = result[i].root[3] - rect.y;
	  cvSetImageROI( img, rect );
	  sprintf(path, "/home/chenqi/workspace/caffe-master/data/Pointing/neg/add2/%d/%d.jpg", cam_num, count_name);
	  count_name++;
	  cvSaveImage(path, img );
	  cvResetImageROI( img );
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

	video_name = result_file.substr( result_file.find_last_of('/')+1, result_file.length()-result_file.find_last_of('/') ); 
	video_name = video_name.substr( 0, video_name.length()-11 );
	cam_num = video_name[19] - '0';
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
			if ( fid%10 == 0)
				save_neg_image( image, result, num );
		
			delete [] result;

		} 
		if (count_name > 4500)
			break;
    }
    fclose( fptr );
    cvReleaseCapture( &capture );
	return 0;
}