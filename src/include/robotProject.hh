#ifndef ROBOTPROJECT
#define ROBOTPROJECT

#include <utility>
#include <utils.hh>
#include <detection.hh>
#include <unwrapping.hh>
#include <calibration.hh>
#include <planning.hh>
#include <configure.hh>
#include <settings.hh>

#include <iostream>
#include "path.h"

using namespace std;
using namespace cv;

class RobotProject{
    public:
        RobotProject(CameraCapture* camera, double& frame_time);
        // RobotProject(int argc, char* argv[]);
        ~RobotProject();

        bool preprocessMap(const Mat & img);
        bool planPath(const Mat & img, Path & path);
        bool localize(const Mat & img, vector<double> & state); //it must execute under 50ms
};

#endif
