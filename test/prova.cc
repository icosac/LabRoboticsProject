#include<iostream>
#include"camera_capture.h"
#include"opencv2/imgproc.hpp"
#include"opencv2/highgui.hpp"
#include"filter.hh"
#include"settings.hh"

using namespace std;
using namespace cv;

void my_imshow(const char* win_name, cv::Mat img, bool reset/*=false*/);
void mywaitkey();

/** Function Headers */
void on_low_h_thresh_trackbar(int, void *);
void on_high_h_thresh_trackbar(int, void *);
void on_low_s_thresh_trackbar(int, void *);
void on_high_s_thresh_trackbar(int, void *);
void on_low_v_thresh_trackbar(int, void *);
void on_high_v_thresh_trackbar(int, void *);

#define FILE_NAME "data/settings.xml"

/** Global Variables */
int low_h=30, low_s=30, low_v=30;
int high_h=100, high_s=100, high_v=100;
Filter filter = Filter(low_h, low_s, low_v, high_h, high_s, high_v);


int main (){
	Settings* s=new Settings();
	cout << *s << endl;
	return 0;
	//Create camera object
	CameraCapture::input_options_t options(1080, 1920, 30, 0);
	CameraCapture *camera=new CameraCapture(options);	

	Mat frame;
	double time;
	if (camera->grab(frame, time)){
		cout << "Success" << endl;
	}
	else {
		cout << "Fail" << endl;
	}
	my_imshow("Frame", frame, false);
	mywaitkey();

	cout << "Is this ok? [Y/n]  ";
	char choice='n';
	cin >> choice;
	if (choice=='n' || choice=='N'){
    	Mat frame_threshold; 
    	cvtColor(frame, frame, COLOR_BGR2HSV);
    	namedWindow("Filtered Image", WINDOW_NORMAL);
		
		//-- Trackbars to set thresholds for RGB values
	    createTrackbar("Low H","Filtered Image", &low_h, 180, on_low_h_thresh_trackbar);
	    createTrackbar("High H","Filtered Image", &high_h, 180, on_high_h_thresh_trackbar);
	    createTrackbar("Low S","Filtered Image", &low_s, 255, on_low_s_thresh_trackbar);
	    createTrackbar("High S","Filtered Image", &high_s, 255, on_high_s_thresh_trackbar);
	    createTrackbar("Low V","Filtered Image", &low_v, 255, on_low_v_thresh_trackbar);
	    createTrackbar("High V","Filtered Image", &high_v, 255, on_high_v_thresh_trackbar);
	    
		//BLACK FILTER
		//TODO use filters to set global variables
		cout << "Black filter. " << endl;
	    while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
		cout << "Black filter done: " << filter << endl;

	    //RED FILTER
		//TODO use filters to set global variables
	    cout << "Red filter. " << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
		cout << "Red filter done: " << filter << endl;
	    
	    //GREEN FILTER
		//TODO use filters to set global variables
	    cout << "Green filter. " << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
		cout << "Green filter done: " << filter << endl;
	    
	    //VICTIMS FILTER
		//TODO use filters to set global variables
	    cout << "Victims filter. " << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
		cout << "Victims filter done: " << filter << endl;
	    
	    //BLUE FILTER
		//TODO use filters to set global variables
	    cout << "Blue filter. " << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
		cout << "Blue filter done: " << filter << endl;
	}

	free(camera);
	return 0;
}

//! [low]
/** @function on_low_h_thresh_trackbar */
void on_low_h_thresh_trackbar(int, void *)
{
    filter.low_h = min(high_h-1, low_h);
    setTrackbarPos("Low H","Filtered Image", filter.low_h);
}
//! [low]
//! [high]
/** @function on_high_h_thresh_trackbar */
void on_high_h_thresh_trackbar(int, void *)
{
    filter.high_h = max(high_h, low_h+1);
    setTrackbarPos("High H", "Filtered Image", filter.high_h);
}
//![high]
/** @function on_low_s_thresh_trackbar */
void on_low_s_thresh_trackbar(int, void *)
{
    filter.low_s = min(high_s-1, low_s);
    setTrackbarPos("Low S","Filtered Image", filter.low_s);
}

/** @function on_high_s_thresh_trackbar */
void on_high_s_thresh_trackbar(int, void *)
{
    filter.high_s = max(high_s, low_s+1);
    setTrackbarPos("High S", "Filtered Image", filter.high_s);
}

/** @function on_low_v_thresh_trackbar */
void on_low_v_thresh_trackbar(int, void *)
{
    filter.low_v= min(high_v-1, low_v);
    setTrackbarPos("Low V","Filtered Image", filter.low_v);
}

/** @function on_high_v_thresh_trackbar */
void on_high_v_thresh_trackbar(int, void *)
{
    filter.high_v = max(high_v, low_v+1);
    setTrackbarPos("High V", "Filtered Image", filter.high_v);
}

void write2File (){
	// FileStorage fs(FILE_NAME, FileStorage::APP);

	// fs.release();
}
















































void my_imshow( const char* win_name, 
                cv::Mat img, 
                bool reset/*=false*/){
    const int SIZE     = 250;
    const int W_0      = 0;
    const int H_0      = 0;
    const int W_OFFSET = 20;
    const int H_OFFSET = 90;
    const int LIMIT    = W_0 + 5*SIZE + 4*W_OFFSET;

    static int W_now = W_0;
    static int H_now = H_0;
    if(reset){
        W_now = W_0;
        H_now = H_0;
    }

    //const string s = win_name;
    #if CV_MAJOR_VERSION<4
      namedWindow(win_name, CV_WINDOW_NORMAL);
    #else 
      namedWindow(win_name, WINDOW_NORMAL);
    #endif
    
    cvResizeWindow(win_name, SIZE, SIZE);
    imshow(win_name, img);
    moveWindow(win_name, W_now, H_now);
    W_now += SIZE + W_OFFSET;
    if(W_now >= LIMIT){
        W_now = W_0;
        H_now += SIZE + H_OFFSET;
    }
}

void mywaitkey() {
    while((char)waitKey(1)!='q'){}
}