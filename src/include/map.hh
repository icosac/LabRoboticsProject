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
#include <objects.hh>

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

        constexpr static double baseDistance = -1.0;  //It is the reference base distance for the matrix of distances
        const int range = 3;            // It is the foundamental parameter of the function minPath (the right compromise its 3)
        const int foundLimit = 5;       // Empiric limit of found, it represent how many times the destination will be visited before the end of the BFS. 0 is the base case (first visit=stop) ~150, or better, none is the opposite limit.
        const int offsetValue = 50;     // It is the offset applied to the obstacles defined in millimeters (it must contain also the border dimension).
        static const int borderSizeDefault = 4;    // It is the default of the border. The border is defined respect to the size of the cells. The border start from the most external cells of the obstacle and go inside (NOT OUTSIDE ! ! !), this mean that the offset value must contain the correct offset and even the border (off = real_off + border)
        static const int cellSize = 5;  // It is the default size of the each cell: 10x10 pixels
        static const int nPoints = 20;  // It is the number of points that the function sampleNPoints will sample from the computed vector of vector retrieved from the minPath.    

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

        vector<Obstacle> vObstacles;    // It is the vector containing all the obstacles of the map
        vector<Victim> vVictims;        // It is the vector containing all the victims   of the map
        vector<Gate> vGates;            // It is the vector containing all the gates     of the map

    public:
        Mapp(const int _lengthX=1000, const int _lengthY=1500, const int _pixX=cellSize, const int _pixY=cellSize, const int _borderSize=borderSizeDefault, const vector< vector<Point2<int> > > & vvp = vector< vector<Point2<int> > >() );

        void addObject(const vector<Point2<int> > & vp, const OBJ_TYPE type);
            void addObjects(const vector< vector< Point2<int> > > & vvp, const OBJ_TYPE type);
            void addObjects(const vector<Obstacle> & objs);
            void addObjects(const vector<Victim> & objs);
            void addObjects(const vector<Gate> & objs);
            
        void getVictimCenters(vector<Point2<int> > & vp);
        void getGateCenter(vector<Point2<int> > & vp);

        OBJ_TYPE getPointType(const Point2<int> & p);
        bool checkSegment(const Point2<int> & p0, const Point2<int> & p1);
            bool checkSegmentCollisionWithType(const Point2<int> & p0, const Point2<int> & p1, const OBJ_TYPE type);
        
        vector<vector<Point2<int> > > minPathNPoints(const vector<Point2<int> > & vp);
        vector<Point2<int> > minPathTwoPoints(const Point2<int> & p0, const Point2<int> & p1);
        vector<Point2<int> > sampleNPoints(const vector<vector<Point2<int> > > & vvp, const int n=nPoints);
            vector<Point2<int> > sampleNPoints(const vector<Point2<int> > & points, const int n);
            vector<Point2<int> > samplePointsEachNCells(const vector<Point2<int> > & points, const int step);

        Mat createMapRepresentation();
            void imageAddSegments(Mat & map, const vector<Point2<int> > & vp, const int thickness=3);
            void imageAddPoint(Mat & map, const Point2<int> & p, const int radius=7);

        void printMap();
        string matrixToString();
        void printDimensions();
};

#endif