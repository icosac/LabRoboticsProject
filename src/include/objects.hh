#ifndef OBJECTS
#define OBJECTS

#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "clipper.hh"
#include "maths.hh"

using namespace cv;
using namespace std;

class Object{
    public:
        string toString();

        unsigned size();
        unsigned nPoints();

        void computeCenter();
        void computeRadius();
        
        void offsetting(const int offset);
        bool insidePolyApprox(Point2<int> pt);
        bool insidePoly(Point2<int> pt);
        //bool collision(Point2<int> p1, Point2<int> p2);

    protected:
        vector<Point2<int> > points;
        Point2<int> center;
        float radius;
};

class Obstacle: public Object{
    public:
    Obstacle(vector<Point2<int> > vp);

    string toString();
    void print();
};

class Victim: public Object{
    public:
        Victim(vector<Point2<int> > vp, int _value);
        string toString();
        void print();

        /*bool insidePoly(Point2<int> pt){
            return(insidePolyApprox(pt));
        }*/

        int getValue()
            {return(value);}
        void setValue(int v)
            {value=v;}

    protected:
        int value;
};

#endif

//compile command:
//g++ `pkg-config --cflags opencv` -std=c++11 -Wall -O3  -o objects.out clipper.cc objects.cc `pkg-config --libs opencv`