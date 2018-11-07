#ifndef UNWRAPPING_HH
#define UNWRAPPING_HH

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

int unwrapping();

// support function
void loadCoefficients(const string& filename, Mat& camera_matrix, Mat& dist_coeffs);
void my_imshow(const char*  win_name, Mat img, bool reset = false);
float distance(Point c1, Point c2);
void swap(int & a, int & b);

#endif