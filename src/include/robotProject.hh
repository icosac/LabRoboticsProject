#ifndef ROBOTPROJECT
#define ROBOTPROJECT

#include <utils.hh>

#include <iostream>
#include <clipper.hh>

using namespace std;
using namespace cv;

class RobotProject{
    public:
        RobotProject();
        bool preprocessMap(/*const*/Mat & img);
        bool planPath(/*const*/ Mat & img, ClipperLib::Path & path);//path from clipper
        bool localize(/*const*/ Mat & imgNew, vector<double> & state); //it must execute under 50ms
};

#endif
