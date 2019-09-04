#ifndef ROBOTPROJECT
#define ROBOTPROJECT

#include <utils.hh>
#include <detection.hh>
#include <unwrapping.hh>
#include <calibration.hh>
#include <planning.hh>
#include <configure.hh>
#include <settings.hh>

#include <iostream>
#include <clipper.hh>

using namespace std;
using namespace cv;

class RobotProject{
    public:
        RobotProject();
        bool preprocessMap(const Mat & img);
        bool planPath(const Mat & img, ClipperLib::Path & path);
        bool localize(const Mat & img, vector<double> & state); //it must execute under 50ms
};

#endif
