#ifndef CONFIGURE_HH
#define CONFIGURE_HH

#include<iostream>

#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core/core_c.h>

#include<utils.hh>
#include<filter.hh>
#include<camera_capture.hh>
#include<settings.hh>

using namespace std;
using namespace cv;

void configure();
void on_low_h_thresh_trackbar(int, void *);
void on_high_h_thresh_trackbar(int, void *);
void on_low_s_thresh_trackbar(int, void *);
void on_high_s_thresh_trackbar(int, void *);
void on_low_v_thresh_trackbar(int, void *);
void on_high_v_thresh_trackbar(int, void *);
void update_trackers();
bool show_all_conditions(const Mat& frame, Settings* s);

#endif
