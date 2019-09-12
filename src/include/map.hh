#ifndef MAPP
#define MAPP

#include <vector>
#include <set>
#include <queue>
#include <tuple>
#include <iostream>
#include <iomanip>
#include <algorithm>

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

        const int offsetValue = 65;     // It is the offset applied to the obstacles defined in millimeters (it must contain also the border dimension).
        static const int borderSizeDefault = 8;    // It is the default of the border. The border is defined respect to the size of the cells. The border start from the most external cells of the obstacle and go inside (NOT OUTSIDE ! ! !), this mean that the offset value must contain the correct offset and even the border (off = real_off + border)
        static const int cellSize = 5;  // It is the default size of the each cell: 10x10 pixels

        set<pair<int, int> > cellsFromSegment(const Point2<int> & p0, const Point2<int> & p1);

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
        Mapp(   const int _lengthX=1000, const int _lengthY=1500, 
                const int _pixX=cellSize, const int _pixY=cellSize, 
                const int _borderSize=borderSizeDefault, 
                const vector< vector<Point2<int> > > & vvp = vector< vector<Point2<int> > >() );
        ~Mapp();

        void addObject(const vector<Point2<int> > & vp, const OBJ_TYPE type);
            void addObjects(const vector< vector< Point2<int> > > & vvp, const OBJ_TYPE type);
            void addObjects(const vector<Obstacle> & objs);
            void addObjects(const vector<Victim> & objs);
            void addObjects(const vector<Gate> & objs);
            
        void getVictimCenters(vector<Point2<int> > & vp);
        void getGateCenter(vector<Point2<int> > & vp);

        OBJ_TYPE getPointType(const Point2<int> & p);
        OBJ_TYPE getCellType(const int i, const int j);
        bool checkSegment(const Point2<int> & p0, const Point2<int> & p1);
            bool checkSegmentCollisionWithType(const Point2<int> & p0, const Point2<int> & p1, const OBJ_TYPE type);

        Mat createMapRepresentation();
            void imageAddSegments(Mat & image, const vector<Point2<int> > & vp, const int thickness=3);
            void imageAddSegment(Mat & image, const Point2<int> & p0, const Point2<int> & p1, const int thickness);
            void imageAddPoints(Mat & image, const vector<Point2<int> > & vp, const int radius=7);
            void imageAddPoint(Mat & image, const Point2<int> & p, const int radius=7, const Scalar color = Scalar(0, 255, 255));

        void printMap();
        string matrixToString();
        void printDimensions();

        int getOffsetValue() { return this->offsetValue; }
        int getBorderSizeDefault() { return this->borderSizeDefault; }
        int getCellSize() { return this->cellSize; }
        int getLengthX() { return this->lengthX; }
        int getLengthY() { return this->lengthY; }
        int getActualLengthX() { return (this->lengthX-this->offsetValue); }
        int getActualLengthY() { return (this->lengthY-this->offsetValue); }

        int getDimX(){ return this->dimX; }
        int getDimY(){ return this->dimY; }
        int getPixX(){ return this->pixX; }
        int getPixY(){ return this->pixY; }
        int getBorderSize(){ return this->borderSize; }

};

#endif