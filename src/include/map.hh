#ifndef MAPP
#define MAPP

#include <vector>
#include <set>
#include <queue>
#include <tuple>
#include <iostream>

#include <maths.hh>
#include <settings.hh>
#include <utils.hh>
//#include <dubins.hh>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

enum OBJ_TYPE {FREE, VICT, OBST, GATE, BODA/*shortcut for border*/};

class Mapp{
    protected:
        OBJ_TYPE **map;
        const static int baseDistance = 0;
        int **distances;    // neccessary for the min path function distance of the cells from the start cell
        Point2<int> **parents;

        set<pair<int, int> > cellsFromSegment(const Point2<int> p0, const Point2<int> p1);
        void resetDistanceMap(const int value = baseDistance);

        int lengthX;    // dimension of the arena default: 1000
        int lengthY;    // dimension of the arena d: 1500
        int dimX;       // how many cells in the map: 200
        int dimY;       // how many cells in the map: 300
        int pixX;       // dimension of the cell (pixels): 5
        int pixY;       // dimension of the cell (pixels): 5
        int borderSize; // how many cells of the object sides are consider border from outside: 2

    public:
        Mapp(const int _lengthX=1000, const int _lengthY=1500, const int _pixX=5, const int _pixY=5, const int border=2,
            const vector< vector<Point2<int> > > & vvp = vector< vector<Point2<int> > >() );

        void addObjects(const vector< vector< Point2<int> > > & vvp, const OBJ_TYPE type);
            void addObject(const vector<Point2<int> > & vp, const OBJ_TYPE type);

        OBJ_TYPE getPointType(const Point2<int> p);
        bool checkSegment(const Point2<int> p1, const Point2<int> p2);
            bool checkSegmentCollisionWithType(const Point2<int> p0, const Point2<int> p1, const OBJ_TYPE type);
        
        vector<Point2<int> > minPathTwoPoints(const Point2<int> startP, const Point2<int> endP, const bool reset=true);
        vector<Point2<int> > sampleNPoints(const int n, const vector<Point2<int> > & points);

        Mat createMapRepresentation(/*eventually add a vector of bubins*/);

        void printMap();
        string matrixToString();
        void printDimensions();
};

#endif