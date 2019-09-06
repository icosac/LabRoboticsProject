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
#include "path.h"

using namespace std;
using namespace cv;

// extern Settings* sett;

class RobotProject{
    public:
        RobotProject(CameraCapture* camera, double& frame_time);
        RobotProject(int argc, char* argv[]);
        ~RobotProject();

        bool preprocessMap(const Mat & img);
        bool planPath(const Mat & img, Path & path);
        bool localize(const Mat & img, vector<double> & state); //it must execute under 50ms
};

#endif


	// //Throw away first n frames to calibrate camera
	// CameraCapture::input_options_t options(1080, 1920, 30, 0);
 //    camera= new CameraCapture(options);

 //    double frame_time=0.0;
 //    for (int i=0; i<50; i++){
 //        Mat frame;
 //        camera->grab(frame, frame_time);
	//     COUT(frame_time)
 //    }