#ifndef PLANNING_HH
#define PLANNING_HH

/*#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>*/

#include <iostream>
#include <tuple>
#include <vector>

#include "map.hh"
#include "utils.hh"
#include "maths.hh"

using namespace std;
using namespace cv;

//main function
pair< vector<Point2<int> >, Mat > planning();


#endif