#ifndef MAPP
#define MAPP

#include <vector>
#include <iostream>

#include <maths.hh>

enum OBJ_TYPE {FREE, VICT, OBST, GATE};

class Mapp{
    protected:
        OBJ_TYPE **map;

        int lengthX;    // dimension of the arena default: 1000
        int lengthY;    // dimension of the arena d: 1500
        int dimX;       // how many cells in the map: 200
        int dimY;       // how many cells in the map: 300
        int pixX;       // dimension of the cell (pixels): 5
        int pixY;       // dimension of the cell (pixels): 5
        
    public:

        Mapp(int _lengthX=1000, int _lengthY=1500, int _pixX=5, int _pixY=5,
            vector< vector<Point2<int> > > vvp = vector< vector<Point2<int> > >() );

        void addObject(vector<Point2<int> > vp, const OBJ_TYPE type);

        void printMap();
        string matrixToString();
        void printDimensions();

        OBJ_TYPE getPointType(const Point2<int> p);
        bool checkSegment(const Point2<int> p1, const Point2<int> p2);

};

#endif