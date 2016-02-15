#include <stdio.h>  
#include <opencv2\opencv.hpp>
#include <opencv2\stitching\stitcher.hpp>
#include <vector>
#include <string>
#include <Windows.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

 #ifdef _DEBUG    
#pragma comment(lib, "opencv_core2411d.lib")     
#pragma comment(lib, "opencv_imgproc2411d.lib")   //MAT processing    
#pragma comment(lib, "opencv_highgui2411d.lib")    
#pragma comment(lib, "opencv_stitching2411d.lib")

 #else    
#pragma comment(lib, "opencv_core2411.lib")    
#pragma comment(lib, "opencv_imgproc2411.lib")    
#pragma comment(lib, "opencv_highgui2411.lib")    
#pragma comment(lib, "opencv_stitching2411.lib")
 #endif

using namespace cv;
using namespace std;

vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	char search_path[200];
	sprintf(search_path, "%s/*.*", folder.c_str());
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path, &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}


void main()
{
	vector< Mat > vImg;
	Mat rImg;

	auto image_paths = get_all_files_names_within_folder(".");

	for (auto path : image_paths) {
		if (path.find(".jpg") != string::npos) {
			cout << path << endl;
			vImg.push_back(imread(path));
		}
	}

	Stitcher stitcher = Stitcher::createDefault();


	unsigned long AAtime = 0, BBtime = 0; //check processing time  
	AAtime = getTickCount(); //check processing time  

	Stitcher::Status status = stitcher.stitch(vImg, rImg);

	BBtime = getTickCount(); //check processing time   
	printf("%.2lf sec \n", (BBtime - AAtime) / getTickFrequency()); //check processing time  

	if (Stitcher::OK == status) {
		namedWindow("Stitching Result", WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		imshow("Stitching Result", rImg);
		waitKey(0);
	}

	else
		cout << "Stitching failed:" << status << endl;
}