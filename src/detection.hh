#ifndef DETECTION_HH
#define DETECTION_HH

//tesseract
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "utils.hh"

#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

//main function
int detection();

//core function
void shape_detection(Mat img, const int color); //color: 0=red, 1=green, 2=blue, 3=black

void erode_dilation(Mat & img, const int color); //color: 0=red, 1=green, 2=blue, 3=black
void find_contours(const Mat & img, Mat original, const int color);
void save_convex_hull(vector<vector<Point>> & contours, const int color, vector<int> victims={});
int number_recognition(Rect blob, const Mat & base);
void load_number_template();

#endif