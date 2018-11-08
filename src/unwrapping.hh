#ifndef UNWRAPPING_HH
#define UNWRAPPING_HH

#include "utils.hh"
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
void loadCoefficients(  const string filename, 
                        Mat& camera_matrix, 
                        Mat& dist_coeffs);

#endif