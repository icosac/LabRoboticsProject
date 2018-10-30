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
void my_imshow(const char*  win_name, Mat img);
float distance(Point c1, Point c2);

// sorting point function
void PrintPoints(const char *caption, const vector<Point_<int> > & points);
double Orientation(const Point &a, const Point &b, const Point &c);
void Sort4PointsCounterClockwise(vector<Point_<int> > & points);
double CrossProductZ(const Point &a, const Point &b);

#endif