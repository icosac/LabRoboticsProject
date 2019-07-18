#include<iostream>
#include"camera_capture.h"
#include"opencv2/imgproc.hpp"
#include"opencv2/highgui.hpp"
#include"filter.hh"
#include"settings.hh"

using namespace std;
using namespace cv;

void my_imshow(const char* win_name, cv::Mat img, bool reset=false);
void mywaitkey();

/** Function Headers */
void on_low_h_thresh_trackbar(int, void *);
void on_high_h_thresh_trackbar(int, void *);
void on_low_s_thresh_trackbar(int, void *);
void on_high_s_thresh_trackbar(int, void *);
void on_low_v_thresh_trackbar(int, void *);
void on_high_v_thresh_trackbar(int, void *);
void update_trackers();
bool show_all_conditions(Mat& frame, Settings* s);

#define NAME(x) #x
#define FILE_NAME "data/settings.xml"
// #define DEBUG

/** Global Variables */
// int low_h=30, low_s=30, low_v=30;
// int high_h=100, high_s=100, high_v=100;
Filter filter = Filter(30, 30, 30, 100, 100, 100);

int main (){
	Settings* s=new Settings();
	s->readFromFile();
	cout << *s << endl;

	//Create camera object
	CameraCapture::input_options_t options(1080, 1920, 30, 0);
	CameraCapture *camera=new CameraCapture(options);	

	Mat frame;
	double time;
	if (camera->grab(frame, time)){
		cout << "Success" << endl;
	}
	else {
		cout << "Fail getting camera photo." << endl;
		return 0;
	}
	#ifdef DEBUG
		my_imshow("Frame", frame, false);
		mywaitkey();
	#endif

	if (show_all_conditions(frame, s)){
    	Mat frame_threshold; 
    	cvtColor(frame, frame, COLOR_BGR2HSV);
    	namedWindow("Filtered Image", WINDOW_NORMAL);
		
		//-- Trackbars to set thresholds for RGB values
	    createTrackbar("Low H","Filtered Image", &(filter.low_h), 180, on_low_h_thresh_trackbar);
	    createTrackbar("Low S","Filtered Image", &(filter.low_s), 255, on_low_s_thresh_trackbar);
	    createTrackbar("Low V","Filtered Image", &(filter.low_v), 255, on_low_v_thresh_trackbar);
	    createTrackbar("High H","Filtered Image", &(filter.high_h), 180, on_high_h_thresh_trackbar);
	    createTrackbar("High S","Filtered Image", &(filter.high_s), 255, on_high_s_thresh_trackbar);
	    createTrackbar("High V","Filtered Image", &(filter.high_v), 255, on_high_v_thresh_trackbar);
	    
		//BLACK FILTER
		filter=s->blackMask; update_trackers();
		cout << "Black filter. " << filter << endl;
	    while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image", frame_threshold);
	    }
	    s->changeMask(Settings::BLACK, filter);
		cout << "Black filter done: " << filter << endl;

	    //RED FILTER
		filter=s->redMask; update_trackers();
	    cout << "Red filter. " << filter << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
	    s->changeMask(Settings::RED, filter);
		cout << "Red filter done: " << filter << endl;
	    
	    //GREEN FILTER
		filter=s->greenMask; update_trackers();
	    cout << "Green filter. " << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
	    s->changeMask(Settings::GREEN, filter);
		cout << "Green filter done: " << filter << endl;
	    
	    //VICTIMS FILTER
		filter=s->victimMask; update_trackers();
	    cout << "Victim filter. " << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
	    s->changeMask(Settings::VICTIMS, filter);
		cout << "Victims filter done: " << filter << endl;
	    
	    //BLUE FILTER
		filter=s->blueMask; update_trackers();
	    cout << "Blue filter. " << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }

	    //WHITE FILTER
		filter=s->whiteMask; update_trackers();
	    cout << "White filter. " << endl;
		while ((char)waitKey(1)!='c'){
	    	inRange(frame, filter.Low(), filter.High(), frame_threshold);
	    	//-- Show the frames
        	imshow("Filtered Image",frame_threshold);
	    }
	    s->changeMask(Settings::WHITE, filter);
		cout << "White filter done: " << filter << endl;
	}
	
	s->writeToFile();

	free(camera);
	return 0;
}

//TODO can't find bug
bool show_all_conditions(Mat& frame, Settings* s){
	Mat black, red, green, victim, blue, white;
	cvtColor(frame, frame, COLOR_BGR2GRAY);

	namedWindow (NAME(frame), WINDOW_NORMAL);
	// namedWindow (NAME(black), WINDOW_NORMAL);
	// namedWindow (NAME(red), WINDOW_NORMAL);
	// namedWindow (NAME(green), WINDOW_NORMAL);
	// namedWindow (NAME(victim), WINDOW_NORMAL);
	// namedWindow (NAME(blue), WINDOW_NORMAL);
	// namedWindow (NAME(white), WINDOW_NORMAL);
	imshow(NAME(frame), frame);

	// inRange(frame, s->blackMask.Low(), s->blackMask.High(), black);
	// inRange(frame, s->redMask.Low(), s->redMask.High(), red);
	// inRange(frame, s->greenMask.Low(), s->greenMask.High(), green);
	// inRange(frame, s->victimMask.Low(), s->victimMask.High(), victim);
	// inRange(frame, s->blueMask.Low(), s->blueMask.High(), blue);
	// inRange(frame, s->whiteMask.Low(), s->whiteMask.High(), white);
	
	// imshow(NAME(black), black);
	// my_imshow("red", red);
	// my_imshow("green", green, true);
	// my_imshow("victim", victim);
	// my_imshow("blue", blue);
	// my_imshow("white", white);

	while(true){
		char choice='a';
		cout << "Is this ok? [Y/n]" << endl;
		cin >> choice;
		if (choice=='y' || choice=='Y'){
			return false;
		}
		else if (choice=='n' || choice=='N'){
			return true;
		}
	} 

}

void update_trackers(){
	setTrackbarPos("Low H","Filtered Image", filter.low_h);
	setTrackbarPos("Low S","Filtered Image", filter.low_s);
	setTrackbarPos("Low V","Filtered Image", filter.low_v);
	setTrackbarPos("High H","Filtered Image", filter.high_h);
	setTrackbarPos("High S","Filtered Image", filter.high_s);
	setTrackbarPos("High V","Filtered Image", filter.high_v);
}

//! [low]
/** @function on_low_h_thresh_trackbar */
void on_low_h_thresh_trackbar(int, void *)
{
    filter.low_h = min(filter.high_h-1, filter.low_h);
    setTrackbarPos("Low H","Filtered Image", filter.low_h);
}
//! [low]
//! [high]
/** @function on_high_h_thresh_trackbar */
void on_high_h_thresh_trackbar(int, void *)
{
    filter.high_h = max(filter.high_h, filter.low_h+1);
    setTrackbarPos("High H", "Filtered Image", filter.high_h);
}
//![high]
/** @function on_low_s_thresh_trackbar */
void on_low_s_thresh_trackbar(int, void *)
{
    filter.low_s = min(filter.high_s-1, filter.low_s);
    setTrackbarPos("Low S","Filtered Image", filter.low_s);
}

/** @function on_high_s_thresh_trackbar */
void on_high_s_thresh_trackbar(int, void *)
{
    filter.high_s = max(filter.high_s, filter.low_s+1);
    setTrackbarPos("High S", "Filtered Image", filter.high_s);
}

/** @function on_low_v_thresh_trackbar */
void on_low_v_thresh_trackbar(int, void *)
{
    filter.low_v= min(filter.high_v-1, filter.low_v);
    setTrackbarPos("Low V","Filtered Image", filter.low_v);
}

/** @function on_high_v_thresh_trackbar */
void on_high_v_thresh_trackbar(int, void *)
{
    filter.high_v = max(filter.high_v, filter.low_v+1);
    setTrackbarPos("High V", "Filtered Image", filter.high_v);
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