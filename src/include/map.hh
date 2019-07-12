#ifndef MAP
#define MAP

#include <vector>
#include <iostream>

#include "maths.hh"

class Map{
    public:
        typedef short int bit;  // it can be chenged to bool

        Map(int lengthX=1000, int lengthY=1500, int xx=5, int yy=5,
            vector< vector<Point2<int> > > vvp = vector< vector<Point2<int> > >() );

        void addObject(vector<Point2<int> > vp);
        void printMatrix();
        string toString();

        bool checkPoint(Point2<int> p);
        bool checkSegment(Point2<int> p1, Point2<int> p2);

    protected:
        bit **map;

        int lengthX;    // dimension of the arena default: 1000
        int lengthY;    // dimension of the arena d: 1500
        int dimX;       // how many cells in the map: 200
        int dimY;       // how many cells in the map: 300
        int xx;         // dimension of the cell (pixels): 5
        int yy;         // dimension of the cell (pixels): 5
};

#endif