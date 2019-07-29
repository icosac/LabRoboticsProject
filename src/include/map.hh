#ifndef MAPP
#define MAPP

#include <vector>
#include <set>
#include <queue>
#include <tuple>
#include <iostream>
#include <iomanip>

#include <maths.hh>
#include <settings.hh>
#include <utils.hh>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

enum OBJ_TYPE {FREE, VICT, OBST, GATE, BODA/*shortcut for border*/};

class Mapp{
    protected:
        OBJ_TYPE **map;
        constexpr static double baseDistance = -1.0;  //it is the reference base distance for the matrix of distances

        set<pair<int, int> > cellsFromSegment(const Point2<int> & p0, const Point2<int> & p1);
        vector<Point2<int> > minPathTwoPointsInternal(
                                const Point2<int> & startP, const Point2<int> & endP, 
                                double ** distances, Point2<int> ** parents);
        void resetDistanceMap(double ** distances, const double value = baseDistance);

        int lengthX;    // dimension of the arena default: 1000
        int lengthY;    // dimension of the arena d: 1500
        int dimX;       // how many cells in the map: 200
        int dimY;       // how many cells in the map: 300
        int pixX;       // dimension of the cell (pixels): 5
        int pixY;       // dimension of the cell (pixels): 5
        int borderSize; // how many cells of the object sides are consider border from outside: 2

    public:
        Mapp(const int _lengthX=1000, const int _lengthY=1500, const int _pixX=10, const int _pixY=10, const int border=2,
            const vector< vector<Point2<int> > > & vvp = vector< vector<Point2<int> > >() );

        void addObjects(const vector< vector< Point2<int> > > & vvp, const OBJ_TYPE type);
            void addObject(const vector<Point2<int> > & vp, const OBJ_TYPE type);

        OBJ_TYPE getPointType(const Point2<int> & p);
        bool checkSegment(const Point2<int> & p0, const Point2<int> & p1);
            bool checkSegmentCollisionWithType(const Point2<int> & p0, const Point2<int> & p1, const OBJ_TYPE type);
        
        vector<vector<Point2<int> > > minPathNPoints(const vector<Point2<int> > & vp);
        vector<Point2<int> > minPathTwoPoints(const Point2<int> & p0, const Point2<int> & p1);
        vector<Point2<int> > sampleNPoints(const int n, const vector<Point2<int> > & points);

        Mat createMapRepresentation();
            void imageAddSegments(Mat & map, const vector<Point2<int> > & vp, const int thickness=3);
            void imageAddPoint(Mat & map, const Point2<int> & p, const int radius=7);

        void printMap();
        string matrixToString();
        void printDimensions();
};

#endif